use reqwest::{self, StatusCode};
use thorium::{
    client::{self, Users},
    models::{AuthResponse, UserCreate, UserRole, UserUpdate},
    Error,
};

use crate::k8s::clusters::ClusterMeta;
use crate::k8s::secrets;

/// Create a thorium user account or auth if the account exists
///
/// The operator often needs to create a new admin user accounts. For existing clusters
/// configured initially without an operator, the user accounts may already exist but with
/// passwords that the operator does not know. The force option when passed in with an
/// admin user token will optionally allow the operator to override and update passwords
/// for existing users.
///
/// # Arguments
///
/// * `meta` - Thorium cluster client and metadata
/// * `url` - The Thorium API URL passed to the operator as an argument
/// * `username` - Name of the user
/// * `password` - Password for the user
/// * `admin` - Should the user be an admin
/// * `thorium` - Thorium API client to use for forced password updates
pub async fn create_or_auth_user(
    meta: &ClusterMeta,
    url: &str,
    username: &str,
    password: &str,
    admin: bool,
    thorium: Option<&thorium::Thorium>,
) -> Result<AuthResponse, Error> {
    let client = reqwest::Client::new();
    // build out user request, make user a local auth user with admin role
    let mut user_req = UserCreate::new(username, password, "thorium").skip_verification();
    // set local auth to true in case cluster is LDAP enabled
    user_req.local = true;
    // we want users created by the operator to be admin
    user_req.role = UserRole::Admin;
    let settings = client::ClientSettings::default();
    // key is none for non-admin users
    let mut key: Option<String> = None;
    // pass in thorium secret_key if creating an admin account
    if admin {
        key = Some(meta.cluster.spec.config.thorium.secret_key.clone());
    }
    // attempt to create the user account if it doesn't exist
    let result = thorium::client::Users::create(url, user_req, key.as_deref(), &settings).await;
    // user was created
    match result {
        Ok(auth_result) => {
            println!("Created {} user", username);
            Ok(auth_result)
        }
        Err(error) => {
            match error.status() {
                Some(StatusCode::CONFLICT) => {
                    println!("User {} already exists", username);
                    // force override of old password, used when we don't have a password secret
                    // to grab the user password from
                    match thorium {
                        Some(thorium_api) => {
                            println!("Attempting force reset of password with an admin user token");
                            let update = UserUpdate {
                                password: Some(password.to_owned()),
                                email: None,
                                role: if admin { Some(UserRole::Admin) } else { None },
                                settings: None,
                            };
                            // update the user via the Thorium client
                            thorium_api.users.update(username, update).await?;
                            println!("Password reset successful for {}", username);
                            println!("Attempting basic auth with {}'s password", username);
                            // attempt basic auth with password and return AuthResponse
                            Users::auth_basic(url, username, &password, &client).await
                        }
                        // user exists and no admin token was provided, lets just auth with user's pass
                        None => {
                            println!("Attempting basic auth with {}'s password", username);
                            // attempt basic auth with password and return AuthResponse
                            Users::auth_basic(url, username, &password, &client).await
                        }
                    }
                }
                _ => Err(Error::new(format!(
                    "Failed to create {} user: {}",
                    username, error
                ))),
            }
        }
    }
}

/// Create an operator user account
///
/// The operator user account is used to configure all other admin accounts and its token
/// is used for subsequent admin actions such as adding nodes and initializing system
/// settings. If this method fails, further configuration of the ThoriumCluster cannot
/// continue. This returns the token of the operator user if account creation/auth was a
/// success. The operator user can be an existing account so long as there is a
/// corresponding kubernetes secret that contains the user's password.
///
/// # Arguments
///
/// * `meta` - Thorium cluster client and metadata
/// * `url` - The Thorium API URL passed to the operator as an argument
pub async fn create_operator(meta: &ClusterMeta, url: &str) -> Result<String, Error> {
    let username = "thorium-operator";
    // try and create secret if does not exist
    let mut password = secrets::create_user_secret(username, meta).await?;
    if password.is_none() {
        // grab existing password from secret
        println!("Operator secret exists, not updating password");
        password = secrets::get_user_password(username, meta).await?;
    }
    // create the operator user: thorium-operator
    match password {
        Some(operator_pass) => {
            println!("Creating operator user");
            let auth = create_or_auth_user(
                meta,
                url,
                "thorium-operator",
                operator_pass.as_ref(),
                true,
                None,
            )
            .await?;
            // return token so operator user can be used to operate stuff
            Ok(auth.token)
        }
        None => Err(Error::new(format!(
            "Could not generate new or get existing operator password"
        ))),
    }
}

/// Create a Thorium admin account
///
/// Create a thorium admin account such as thorium or thorium-kaboom. These accounts
/// will have their password added to keys.yml secrets for use by the agents and any
/// scalers.
///
/// # Arguments
///
/// * `meta` - Thorium cluster client and metadata
/// * `thorium` - Thorium API client to use for forced password updates
/// * `url` - The Thorium API URL passed to the operator as an argument
/// * `token` - Admin token to use for overriding user passwords
/// * `username` - Name of user to create
pub async fn create(
    meta: &ClusterMeta,
    thorium: &thorium::Thorium,
    url: &str,
    username: &str,
) -> Result<(String, String), Error> {
    // try and create secret; otherwise, grab existing secret
    let mut password = secrets::create_user_secret(username, meta).await?;
    if password.is_none() {
        println!("{} user secret exists, not updating password", username);
        password = secrets::get_user_password(username, meta).await?;
    }
    // create the operator user: thorium-operator
    match password {
        Some(user_pass) => {
            println!("Creating {} user", username);
            let auth =
                create_or_auth_user(meta, url, username, user_pass.as_ref(), true, Some(thorium))
                    .await?;
            // return token so operator user can be used to operate stuff
            Ok((user_pass, auth.token))
        }
        None => Err(Error::new(format!(
            "Could not generate new or get existing {username} user password"
        ))),
    }
}
