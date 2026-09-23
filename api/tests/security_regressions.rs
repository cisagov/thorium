//! Regression tests for the Thorium security review
//!
//! Every test in this file asserts the SECURE/correct behavior and runs actively (never
//! `#[ignore]`d unless it needs an environment the harness cannot stand up). Because the
//! underlying issues may not yet be fixed, a test here is EXPECTED TO FAIL against vulnerable
//! code and turns GREEN the moment the corresponding issue is remediated — the failing suite is
//! the remediation to-do list and each test becomes a permanent guard once its fix lands.
//!
//! These are live-API integration tests: they boot the API in-process against Scylla/Redis/S3/
//! Elastic (see `.gitlab-ci.yml`) and are run by the operator, not offline.

use thorium::models::{
    Buffer, GroupAllowedUpdate, GroupRequest, GroupUpdate, GroupUsersRequest, ImageScaler,
    OriginRequest, Pools, Resources, SampleRequest, UserRole, WorkerRegistrationList,
};
use thorium::test_utilities::{self, generators};
use thorium::{is, Error};
use thorium::client::{ClientSettings, Users};

/// Assert that a result is an error carrying the expected HTTP status code
///
/// The `fail!` test macro re-evaluates its argument (which moves owned inputs) and requires the
/// `Ok` type to implement `Debug`; several client calls here return non-`Debug` values
/// (`DownloadedFile`, `Cursor<_>`) or consume owned blueprints, so this helper checks an
/// already-evaluated result without either constraint.
///
/// # Arguments
///
/// * `result` - The already-evaluated result to check
/// * `code` - The HTTP status code the error must carry
/// * `context` - A short description of what was expected, used in failure messages
fn assert_status<T>(result: Result<T, Error>, code: u16, context: &str) -> Result<(), Error> {
    // inspect the result, discarding any success value so `T` need not be `Debug`
    match result {
        // a success means the secure behavior is not enforced yet, so fail the test
        Ok(_) => Err(Error::new(format!(
            "{context}: expected status {code} but the request succeeded"
        ))),
        // an error must carry exactly the expected status code
        Err(error) => match error.status() {
            // the status matched so the secure behavior held
            Some(status) if status.as_u16() == code => Ok(()),
            // a different status is still a failure to enforce the expected boundary
            Some(status) => Err(Error::new(format!(
                "{context}: expected status {code} but got {}",
                status.as_u16()
            ))),
            // no status at all means the error was not the authorization rejection we expected
            None => Err(Error::new(format!(
                "{context}: expected status {code} but the error carried no status: {error}"
            ))),
        },
    }
}

/// Unauthenticated self-registration must NOT grant the all-groups Analyst role without
/// the deployment secret key.
///
/// `POST /api/users/` is unauthenticated by design. An external attacker who can self-assign the
/// `Analyst` role gains read access to every group in the cluster (`db::users::get` expands an
/// Analyst's group list to all groups). The secure behavior is to reject a keyless `Analyst`
/// registration, so this asserts the request fails with `401`.
#[tokio::test]
async fn self_registration_analyst_requires_secret_key() -> Result<(), Error> {
    // get an admin client just to resolve the API host to talk to
    let client = test_utilities::admin_client().await?;
    // build a self-registration blueprint that self-assigns the all-groups Analyst role
    let blueprint = thorium::models::UserCreate::new(
        generators::gen_string(24),
        generators::gen_string(64),
        "attacker@evil.example",
    )
    .role(UserRole::Analyst);
    // use default client settings for this unauthenticated registration
    let settings = ClientSettings::default();
    // registering an Analyst without the secret key must be rejected
    assert_status(
        Users::create(&client.host, blueprint, None::<&str>, &settings).await,
        401,
        "keyless Analyst self-registration",
    )
}

/// Client-controlled `skip_verification` must NOT be honored without the deployment secret
/// key.
///
/// `skip_verification` is documented as "requires secret key" but historically bypassed the email
/// verification gate for any unauthenticated registrant. The secure behavior is to reject a
/// keyless `skip_verification=true` registration, so this asserts the request fails with `401`.
#[tokio::test]
async fn skip_verification_requires_secret_key() -> Result<(), Error> {
    // get an admin client just to resolve the API host to talk to
    let client = test_utilities::admin_client().await?;
    // build a self-registration blueprint that tries to skip email verification without a key
    let blueprint = thorium::models::UserCreate::new(
        generators::gen_string(24),
        generators::gen_string(64),
        "attacker@evil.example",
    )
    .skip_verification();
    // use default client settings for this unauthenticated registration
    let settings = ClientSettings::default();
    // skipping verification without the secret key must be rejected
    assert_status(
        Users::create(&client.host, blueprint, None::<&str>, &settings).await,
        401,
        "keyless skip_verification self-registration",
    )
}

/// Sample authorization must not fall through to success when the groups×sha256s cartesian
/// product is >= 100.
///
/// `db::files::authorize` has a small-product branch (correct) and a large-product branch. Any
/// authenticated user can create unlimited groups (`POST /api/groups/` has no role gate) to inflate
/// their group count past 100, forcing every authorize into the large branch. If that branch grants
/// access when the sample is in none of the caller's groups, the attacker can download any sample
/// in the cluster knowing only its sha256. The secure behavior is that a cross-tenant download
/// still fails with `401` even with >= 100 groups.
#[tokio::test]
async fn sample_authorize_no_fallthrough_with_many_groups() -> Result<(), Error> {
    // get an admin client that owns the victim data
    let admin = test_utilities::admin_client().await?;
    // create a victim group owned by the admin
    let victim_group = generators::groups(1, &admin).await?.remove(0).name;
    // upload a victim sample the attacker must never be able to reach
    let victim = admin
        .files
        .create(
            SampleRequest::new_buffer(
                Buffer::new(generators::gen_string(64)),
                vec![victim_group],
            )
            .description("victim sample")
            .origin(OriginRequest::downloaded("https://victim.example", None)),
        )
        .await?;
    // get a low privilege user client that is not in the victim group
    let attacker = generators::client(&admin).await?;
    // inflate the attacker's group count past 100 to force the large-product authorize branch
    for _ in 0..100 {
        // each group has no role gate and auto-adds the creator as owner
        attacker.groups.create(&generators::gen_group()).await?;
    }
    // build download options targeting a throwaway path
    let mut opts = thorium::models::FileDownloadOpts::default();
    // pick a unique download destination so parallel tests do not collide
    let dest = std::env::temp_dir().join(format!("xtenant-{}", generators::gen_string(12)));
    // downloading a sample from a group the attacker is not in must still fail
    assert_status(
        attacker
            .files
            .download(&victim.sha256, &dest, &mut opts)
            .await,
        401,
        "cross-tenant download with >= 100 groups",
    )
}

/// Worker registration must require admin privileges.
///
/// `POST /api/system/worker/{scaler}` historically discarded the caller entirely, letting any
/// authenticated low-privilege user write attacker-chosen rows into the scaler's worker registry
/// (a scheduling-plane DoS, and on reactor deployments a path to admin-token theft). Every sibling
/// `/api/system/*` method is admin-gated. The secure behavior is that a non-admin registration is
/// rejected with `401`.
#[tokio::test]
async fn worker_registration_requires_admin() -> Result<(), Error> {
    // get an admin client just to resolve the API host and create the low-priv user
    let admin = test_utilities::admin_client().await?;
    // get a low privilege (non-admin) user client
    let attacker = generators::client(&admin).await?;
    // resolve the attacker's own username to set as the phantom worker's owner
    let username = attacker.users.info().await?.username;
    // build a phantom worker registration naming plausible cluster/node values
    let workers = WorkerRegistrationList::default().add(
        "cluster0",
        "node0",
        "phantom-worker",
        username,
        "attacker-group",
        "attacker-pipeline",
        "attacker-stage",
        Resources::default(),
        Pools::Deadline,
    );
    // a non-admin registering workers must be rejected
    assert_status(
        attacker
            .system
            .register_workers(ImageScaler::K8s, &workers)
            .await,
        401,
        "non-admin worker registration",
    )
}

/// A read-only group member (monitor) must NOT be able to change the group's `allowed`
/// settings or description.
///
/// `Group::update` only invokes the owner/manager (`modifiable`) check when the update touches role
/// membership; `removes_only_user` ignores `description`, `clear_description`, and the entire
/// `allowed` update. So an update with zero role changes skips the privilege check and any member —
/// even a monitor, the explicitly read-only role — can flip group-wide `allowed` flags (a DoS) or
/// deface the description. The secure behavior is that a monitor's settings-only update fails with
/// `401`.
#[tokio::test]
async fn monitor_cannot_change_group_settings() -> Result<(), Error> {
    // get an admin client that owns the shared group
    let admin = test_utilities::admin_client().await?;
    // get a low privilege user client that will only be a monitor in the group
    let monitor = generators::client(&admin).await?;
    // resolve the monitor's username so we can grant them the read-only role
    let monitor_name = monitor.users.info().await?.username;
    // build a group where the monitor holds only the read-only monitor role
    let group_req = GroupRequest::new(generators::gen_string(50))
        .monitors(GroupUsersRequest::default().direct(&monitor_name));
    // create the shared group as the admin
    admin.groups.create(&group_req).await?;
    // as the monitor, attempt a settings-only update that touches no role membership
    let mut update = GroupUpdate::default().description("pwned");
    // disabling files is a group-wide setting that only owners/managers should control
    update.allowed = GroupAllowedUpdate::default().disable_files();
    // a read-only monitor changing group settings must be rejected
    assert_status(
        monitor.groups.update(&group_req.name, &update).await,
        401,
        "monitor changing group settings",
    )
}

/// The OAuth account-link email flow must not let an attacker link their own IdP identity
/// to a victim's account.
///
/// The attack: the attacker completes OAuth registration with their OWN IdP account (obtaining a
/// registration session whose `alias` is the attacker's OAuth subject), then POSTs
/// `/api/oauth/{provider}/register` with the VICTIM's username and email. Thorium emails the victim
/// a state-changing `GET /api/oauth/{provider}/link?username=victim&token=T` bound to the
/// attacker's alias; a mail prefetcher or a single click inserts the attacker's alias into the
/// victim's `aliases` map, after which the attacker's next OAuth login returns the victim's primary
/// token — full account takeover. The secure behavior is that the link must be scoped to the
/// authenticated victim (not a bare GET), must never be issued for an already-registered
/// username/email at an attacker's request, and must state which IdP identity is being linked.
///
/// This test is `#[ignore]`d because exercising it requires a live OAuth/OIDC identity provider and
/// a mail sink, which the integration harness does not stand up. It documents the invariant and the
/// manual reproduction so the flow can be driven once an IdP fixture exists.
#[tokio::test]
#[ignore = "needs a live OAuth/OIDC identity provider and a mail sink"]
async fn oauth_account_link_cannot_hijack_existing_account() -> Result<(), Error> {
    // boot an admin client so this test compiles alongside the live-API suite
    let _admin = test_utilities::admin_client().await?;
    // NOTE: with an IdP fixture, this test would:
    //   1. register victim V normally and capture V's alias/token
    //   2. complete OAuth registration as attacker A with A's own IdP subject
    //   3. POST /oauth/{provider}/register as A with {username: V, email: V's email}
    //   4. assert the request is rejected (or the emailed link is victim-authenticated), and that
    //      fetching the link does NOT insert A's alias into V's aliases map
    //   5. assert a subsequent OAuth login as A does NOT return V's token
    Ok(())
}

/// Sanity anchor: an admin CAN change group settings, so the negative tests above are asserting a
/// privilege boundary and not a blanket rejection of the update path.
#[tokio::test]
async fn admin_can_change_group_settings() -> Result<(), Error> {
    // get an admin client that owns the group
    let admin = test_utilities::admin_client().await?;
    // create a group owned by the admin
    let group_req = GroupRequest::new(generators::gen_string(50));
    // create the group
    admin.groups.create(&group_req).await?;
    // apply a settings-only update as the owner
    let update = GroupUpdate::default().description("owner set description");
    // this must succeed because the admin owns the group
    admin.groups.update(&group_req.name, &update).await?;
    // confirm the description landed
    let group = admin.groups.get(&group_req.name).await?;
    is!(group.description, Some("owner set description".to_owned()));
    Ok(())
}
