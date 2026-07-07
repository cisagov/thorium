//! Test tree routes

use rand::RngCore;
use std::collections::HashMap;
use thorium::test_utilities::{self, generators};
use thorium::{Thorium, contains, contains_key, is};
use uuid::Uuid;

use thorium::models::{
    AssociationKind, AssociationRequest, AssociationTarget, Buffer, Confidence, Entity,
    EntityKinds, EntityListOpts, EntityMetadataRequest, EntityRequest, FileSystemEntityBuilder,
    Flag, OriginRequest, Sample, SampleRequest, Tree, TreeGrowQuery, TreeOpts, TreeQuery,
    TreeSupport, WindowsProcessEntity, WindowsProcessTreeEntity,
};

/// Generate a buffer of random bytes so every test run uploads unique samples
fn random_buffer() -> Vec<u8> {
    // build a buffer to fill with random data
    let mut data = vec![0u8; 64];
    // fill our buffer with random data
    rand::rng().fill_bytes(&mut data);
    data
}

/// Generate a root sample request with no parents
///
/// # Arguments
///
/// * `group` - The group this sample should be in
fn gen_root(group: &str) -> SampleRequest {
    // build a sample request with random data so reruns don't collide
    SampleRequest::new_buffer(Buffer::new(random_buffer()), vec![group])
        .description("tree test root")
        .origin(OriginRequest::downloaded(
            "https://tree-tests.thorium",
            Some("tree-tests".to_string()),
        ))
}

/// Generate a child sample request unpacked from a parent sample
///
/// # Arguments
///
/// * `group` - The group this sample should be in
/// * `parent` - The sha256 of the parent this sample was unpacked from
fn gen_child(group: &str, parent: &str) -> SampleRequest {
    // build a sample request with random data that was unpacked from our parent
    SampleRequest::new_buffer(Buffer::new(random_buffer()), vec![group])
        .description("tree test child")
        .origin(OriginRequest::unpacked(parent, Some("tree-tests".to_owned())))
}

/// Upload a number of children unpacked from a single parent sample
///
/// # Arguments
///
/// * `client` - The client to use when creating these samples
/// * `group` - The group these samples should be in
/// * `parent` - The sha256 of the parent to unpack these samples from
/// * `cnt` - The number of children to create
async fn upload_children(
    client: &Thorium,
    group: &str,
    parent: &str,
    cnt: usize,
) -> Result<Vec<String>, thorium::Error> {
    // preallocate a list for our children's sha256s
    let mut sha256s = Vec::with_capacity(cnt);
    // upload each of our children
    for _ in 0..cnt {
        // upload this child sample
        let resp = client.files.create(gen_child(group, parent)).await?;
        // track this child's sha256
        sha256s.push(resp.sha256);
    }
    Ok(sha256s)
}

/// Upload a chain of samples where each sample is unpacked from the one before it
///
/// # Arguments
///
/// * `client` - The client to use when creating these samples
/// * `group` - The group these samples should be in
/// * `root` - The sha256 of the sample to start this chain from
/// * `depth` - The number of descendants to chain under our root
async fn upload_chain(
    client: &Thorium,
    group: &str,
    root: &str,
    depth: usize,
) -> Result<Vec<String>, thorium::Error> {
    // preallocate a list for our chain's sha256s
    let mut sha256s = Vec::with_capacity(depth);
    // track the parent for the next link in our chain
    let mut parent = root.to_owned();
    // upload each link in our chain
    for _ in 0..depth {
        // upload the next link in our chain
        let resp = client.files.create(gen_child(group, &parent)).await?;
        // this link is the parent of the next link
        parent.clone_from(&resp.sha256);
        // track this link's sha256
        sha256s.push(resp.sha256);
    }
    Ok(sha256s)
}

/// Get the tree node hash for a sample
///
/// # Arguments
///
/// * `sha256` - The sha256 of the sample to hash
fn sample_hash(sha256: &String) -> u64 {
    Sample::tree_hash_direct(sha256)
}

/// Get the tree node hash for an entity
///
/// # Arguments
///
/// * `id` - The id of the entity to hash
fn entity_hash(id: &Uuid) -> u64 {
    Entity::tree_hash_direct(id)
}

/// Check if a tree has a branch between two nodes in either direction
///
/// # Arguments
///
/// * `tree` - The tree to look for a branch in
/// * `left` - The hash of one node in this branch
/// * `right` - The hash of the other node in this branch
fn has_branch(tree: &Tree, left: u64, right: u64) -> bool {
    // check the branches keyed on our left node
    if let Some(branches) = tree.branches.get(&left) {
        // check if any branch points to our right node
        if branches.iter().any(|branch| branch.node == right) {
            return true;
        }
    }
    // check the branches keyed on our right node
    if let Some(branches) = tree.branches.get(&right) {
        // check if any branch points to our left node
        if branches.iter().any(|branch| branch.node == left) {
            return true;
        }
    }
    false
}

/// Build a tree query starting from a single sample scoped to a group
///
/// # Arguments
///
/// * `group` - The group to limit this tree too
/// * `sha256` - The sha256 of the sample to start this tree from
fn start_query(group: &str, sha256: &str) -> TreeQuery {
    // start our tree from this sample
    let mut query = TreeQuery::default().sample(sha256);
    // scope this tree to just our test group
    query.groups = vec![group.to_owned()];
    query
}

/// Build a map of entity names to ids for a specific kind of entity in a group
///
/// # Arguments
///
/// * `client` - The client to use when listing entities
/// * `group` - The group to list entities from
/// * `kind` - The kind of entities to list
async fn entity_map(
    client: &Thorium,
    group: &str,
    kind: EntityKinds,
) -> Result<HashMap<String, Uuid>, thorium::Error> {
    // build the opts to list entities of this kind in our group
    let opts = EntityListOpts::default()
        .groups(vec![group.to_owned()])
        .kind(kind);
    // list the details for these entities
    let mut cursor = client.entities.list_details(&opts).await?;
    // build a map of entity names to ids
    let mut map = HashMap::default();
    // crawl this cursor until it is exhausted
    loop {
        // add this page of entities to our map
        for entity in cursor.data.drain(..) {
            map.insert(entity.name, entity.id);
        }
        // stop crawling once our cursor is exhausted
        if cursor.exhausted() {
            break;
        }
        // get the next page of entities
        cursor.refill().await?;
    }
    Ok(map)
}

/// Get an entity's id from a map of entities or error
///
/// # Arguments
///
/// * `map` - The map of entity names to ids to search
/// * `name` - The name of the entity to get
fn get_entity(map: &HashMap<String, Uuid>, name: &str) -> Result<Uuid, thorium::Error> {
    // get this entity's id if it exists
    match map.get(name) {
        Some(id) => Ok(*id),
        None => Err(thorium::Error::new(format!("Missing entity: {name}"))),
    }
}

/// Create a flag entity and flag some data with it
///
/// # Arguments
///
/// * `client` - The client to use when creating this flag
/// * `group` - The group this flag should be in
/// * `name` - The name for this flag entity
/// * `reasoning` - The reason this data was flagged
/// * `target` - The data to flag
async fn flag_entity(
    client: &Thorium,
    group: &str,
    name: &str,
    reasoning: &str,
    target: AssociationTarget,
) -> Result<Uuid, thorium::Error> {
    // build the flag for this suspicious data
    let flag = Flag {
        suspicion: 8,
        confidence: Confidence::Likely,
        content: None,
        reasoning: reasoning.to_owned(),
    };
    // wrap our flag in a metadata request
    let metadata = EntityMetadataRequest::Flag(flag);
    // build the entity request for this flag
    let entity_req = EntityRequest::new(name, metadata, vec![group]);
    // create our flag entity
    let resp = client.entities.create(entity_req).await?;
    // this flag is the source of the flag association
    let source = AssociationTarget::Entity {
        id: resp.id,
        name: name.to_owned(),
    };
    // build the association flagging our target data
    let assoc_req = AssociationRequest::new(AssociationKind::FlagFor, source)
        .target(target)
        .groups(vec![group.to_owned()]);
    // create this association
    client.associations.create(&assoc_req).await?;
    Ok(resp.id)
}

/// Test starting and getting a basic two node tree
///
/// This guards the fundamental tree workflow: uploading a child with an
/// unpacked origin must produce a tree with both samples and a branch
/// between them when starting a tree from the parent. It also guards the
/// `GET /api/trees/:id` route by making sure a saved tree can be fetched
/// again with the same id and nodes.
#[tokio::test]
async fn basic() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload a root sample
    let root = client.files.create(gen_root(&group)).await?.sha256;
    // upload a single child unpacked from our root
    let child = upload_children(&client, &group, &root, 1).await?.remove(0);
    // start a tree from our root sample
    let tree = client
        .trees
        .start(&TreeOpts::default(), &start_query(&group, &root))
        .await?;
    // get the hashes for our expected nodes
    let root_hash = sample_hash(&root);
    let child_hash = sample_hash(&child);
    // make sure our tree contains exactly our root and child
    is!(tree.data_map.len(), 2);
    contains_key!(tree.data_map, &root_hash);
    contains_key!(tree.data_map, &child_hash);
    // make sure our root is an initial node
    contains!(tree.initial, &root_hash);
    // make sure our root and child are linked
    is!(
        has_branch(&tree, root_hash, child_hash),
        true,
        "root <-> child branch"
    );
    // get this tree by id and make sure it matches
    let fetched = client.trees.get(tree.id).await?;
    is!(fetched.id, tree.id);
    is!(fetched.data_map.len(), tree.data_map.len());
    contains_key!(fetched.data_map, &root_hash);
    contains_key!(fetched.data_map, &child_hash);
    Ok(())
}

/// Test building and growing a deep chain of samples
///
/// This guards the depth semantics of trees where each growth ring should
/// expand the tree exactly one level. Starting a 6 deep chain with a limit
/// of 3 must gather only 3 levels below the root and leave the deepest
/// node in `growable`. It then guards the `PATCH /api/trees/:cursor` grow
/// route by growing 3 more levels from that frontier and checking that the
/// grow response is trimmed to only the newly added nodes. Finally it
/// guards against long origin chains being truncated by walking a full
/// depth tree link by link.
#[tokio::test]
async fn deep() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload a root sample
    let root = client.files.create(gen_root(&group)).await?.sha256;
    // upload a chain of 6 descendants under our root
    let chain = upload_chain(&client, &group, &root, 6).await?;
    // build the opts to only grow our tree 3 levels deep
    let opts = TreeOpts::default().limit(3);
    // start a tree from our root sample
    let tree = client
        .trees
        .start(&opts, &start_query(&group, &root))
        .await?;
    // make sure our tree only reached 3 levels below our root
    is!(tree.data_map.len(), 4);
    contains_key!(tree.data_map, &sample_hash(&root));
    contains_key!(tree.data_map, &sample_hash(&chain[0]));
    contains_key!(tree.data_map, &sample_hash(&chain[1]));
    contains_key!(tree.data_map, &sample_hash(&chain[2]));
    // make sure the levels past our depth limit were not gathered
    is!(
        tree.data_map.contains_key(&sample_hash(&chain[3])),
        false,
        "4th level should not be gathered"
    );
    // make sure the frontier of our tree can still be grown
    contains!(tree.growable, &sample_hash(&chain[2]));
    // build a query to grow this tree from its frontier
    let mut grow_query = TreeGrowQuery::default();
    // grow from every growable node in our tree
    for hash in &tree.growable {
        grow_query.add_growable_ref(*hash);
    }
    // grow this tree 3 more levels
    let grown = client.trees.grow(tree.id, &opts, &grow_query).await?;
    // make sure we grew the same tree
    is!(grown.id, tree.id);
    // make sure only the newly grown nodes were returned
    is!(grown.data_map.len(), 3);
    contains_key!(grown.data_map, &sample_hash(&chain[3]));
    contains_key!(grown.data_map, &sample_hash(&chain[4]));
    contains_key!(grown.data_map, &sample_hash(&chain[5]));
    is!(
        grown.data_map.contains_key(&sample_hash(&root)),
        false,
        "already sent nodes should be trimmed"
    );
    // start a full depth tree from our root sample
    let full = client
        .trees
        .start(&TreeOpts::default(), &start_query(&group, &root))
        .await?;
    // make sure our full tree contains the entire chain
    is!(full.data_map.len(), 7);
    // track the parent for each link in our chain
    let mut parent_hash = sample_hash(&root);
    // make sure every link in our chain is connected
    for sha256 in &chain {
        // get the hash for this link
        let child_hash = sample_hash(sha256);
        // make sure this link exists and is connected to its parent
        contains_key!(full.data_map, &child_hash);
        is!(
            has_branch(&full, parent_hash, child_hash),
            true,
            "chain link branch"
        );
        // this link is the parent of the next link
        parent_hash = child_hash;
    }
    Ok(())
}

/// Test building a wide tree with many children under one parent
///
/// This guards against fan out bugs where a single growth ring only
/// gathers some of a node's children (e.g. pagination or iteration bugs
/// that stop after the first item). All 10 children must be gathered in
/// one ring and each must have a branch back to the root.
#[tokio::test]
async fn wide() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload a root sample
    let root = client.files.create(gen_root(&group)).await?.sha256;
    // upload 10 children unpacked from our root
    let children = upload_children(&client, &group, &root, 10).await?;
    // start a tree from our root sample
    let tree = client
        .trees
        .start(&TreeOpts::default(), &start_query(&group, &root))
        .await?;
    // get the hash for our root node
    let root_hash = sample_hash(&root);
    // make sure our tree contains our root and all of its children
    is!(tree.data_map.len(), 11);
    contains_key!(tree.data_map, &root_hash);
    // make sure each child is in our tree and linked to our root
    for sha256 in &children {
        // get the hash for this child
        let child_hash = sample_hash(sha256);
        // make sure this child exists and is connected to our root
        contains_key!(tree.data_map, &child_hash);
        is!(
            has_branch(&tree, root_hash, child_hash),
            true,
            "root <-> child branch"
        );
    }
    Ok(())
}

/// Test building a tree that is both wide and deep
///
/// This guards combined breadth and depth growth where every node in every
/// ring must be fully expanded before the next ring. A tree with 3
/// children per parent across 3 levels (40 nodes) must be gathered
/// completely with every parent/child branch intact, guarding against
/// partially explored rings silently dropping whole subtrees.
#[tokio::test]
async fn wide_and_deep() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload a root sample
    let root = client.files.create(gen_root(&group)).await?.sha256;
    // build 3 levels of samples with 3 children per parent
    let mut levels = Vec::with_capacity(3);
    // the first level of parents is just our root
    let mut parents = vec![root.clone()];
    // build each level of our tree
    for _ in 0..3 {
        // build the next level of our tree
        let mut level = Vec::with_capacity(parents.len() * 3);
        // upload 3 children for each parent in the level above
        for parent in &parents {
            // upload this parent's children
            level.extend(upload_children(&client, &group, parent, 3).await?);
        }
        // this level is the parents of the next level
        parents.clone_from(&level);
        // track this level
        levels.push(level);
    }
    // start a tree from our root sample
    let tree = client
        .trees
        .start(&TreeOpts::default(), &start_query(&group, &root))
        .await?;
    // make sure our tree contains our root and all 3 levels (1 + 3 + 9 + 27)
    is!(tree.data_map.len(), 40);
    contains_key!(tree.data_map, &sample_hash(&root));
    // track the parents of the level we are checking
    let mut parent_level = vec![root.clone()];
    // make sure every node in every level is in our tree and linked to its parent
    for level in &levels {
        // check each node in this level
        for (index, sha256) in level.iter().enumerate() {
            // get the hash for this node
            let child_hash = sample_hash(sha256);
            // get the hash for this node's parent
            let parent_hash = sample_hash(&parent_level[index / 3]);
            // make sure this node exists and is connected to its parent
            contains_key!(tree.data_map, &child_hash);
            is!(
                has_branch(&tree, parent_hash, child_hash),
                true,
                "level branch"
            );
        }
        // this level is the parents of the next level
        parent_level.clone_from(level);
    }
    Ok(())
}

/// Test building a tree through a realistic windows process tree
///
/// This guards entity support in trees where entities are related through
/// associations instead of origins. A memory dump sample linked to a
/// windows process tree entity via a `ProcessTreeIn` association must pull
/// in the process tree entity and every `ChildProcess` linked process
/// entity below it, guarding against association hops being dropped when
/// crawling from samples into entity chains.
#[tokio::test]
async fn process_tree() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload a memory dump sample to hang our process tree on
    let dump = client.files.create(gen_root(&group)).await?.sha256;
    // build a realistic windows process tree
    let mut builder =
        WindowsProcessTreeEntity::builder("tree-tests-process-tree").tool("volatility");
    builder.add_mut(WindowsProcessEntity::new(4).name("System").image_path("System"));
    builder.add_mut(
        WindowsProcessEntity::new(368)
            .parent_pid(4)
            .name("smss.exe")
            .image_path(r"\SystemRoot\System32\smss.exe"),
    );
    builder.add_mut(
        WindowsProcessEntity::new(444)
            .parent_pid(368)
            .name("csrss.exe")
            .image_path(r"C:\Windows\System32\csrss.exe"),
    );
    builder.add_mut(
        WindowsProcessEntity::new(520)
            .parent_pid(368)
            .name("winlogon.exe")
            .image_path(r"C:\Windows\System32\winlogon.exe"),
    );
    builder.add_mut(
        WindowsProcessEntity::new(3212)
            .parent_pid(520)
            .name("explorer.exe")
            .image_path(r"C:\Windows\explorer.exe"),
    );
    builder.add_mut(
        WindowsProcessEntity::new(4444)
            .parent_pid(3212)
            .name("cmd.exe")
            .image_path(r"C:\Windows\System32\cmd.exe")
            .command("cmd.exe /c powershell.exe"),
    );
    builder.add_mut(
        WindowsProcessEntity::new(4652)
            .parent_pid(4444)
            .name("powershell.exe")
            .image_path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
            .command("powershell.exe -enc SQBFAFgA"),
    );
    // the groups to build this process tree in
    let groups = vec![group.clone()];
    // create our process tree entities
    let tree_id = builder.create_all(&groups, &client).await?;
    // our memory dump is the source of the process tree association
    let source = AssociationTarget::File(dump.clone());
    // our process tree entity is the target of the process tree association
    let target = AssociationTarget::Entity {
        id: tree_id,
        name: "tree-tests-process-tree".to_owned(),
    };
    // build the association linking our process tree to our memory dump
    let assoc_req = AssociationRequest::new(AssociationKind::ProcessTreeIn, source)
        .target(target)
        .groups(groups.clone());
    // create this association
    client.associations.create(&assoc_req).await?;
    // start a tree from our memory dump
    let tree = client
        .trees
        .start(&TreeOpts::default(), &start_query(&group, &dump))
        .await?;
    // resolve our process entities to their ids
    let processes = entity_map(&client, &group, EntityKinds::WindowsProcess).await?;
    // make sure our tree contains our dump, process tree, and all 7 processes
    is!(tree.data_map.len(), 9);
    contains_key!(tree.data_map, &sample_hash(&dump));
    contains_key!(tree.data_map, &entity_hash(&tree_id));
    // make sure our process tree is linked to our memory dump
    is!(
        has_branch(&tree, sample_hash(&dump), entity_hash(&tree_id)),
        true,
        "dump <-> process tree branch"
    );
    // the parent -> child process chains in our process tree
    let chains = [
        ("System", "smss.exe"),
        ("smss.exe", "csrss.exe"),
        ("smss.exe", "winlogon.exe"),
        ("winlogon.exe", "explorer.exe"),
        ("explorer.exe", "cmd.exe"),
        ("cmd.exe", "powershell.exe"),
    ];
    // make sure our root process is linked to our process tree entity
    let system_id = get_entity(&processes, "System")?;
    is!(
        has_branch(&tree, entity_hash(&tree_id), entity_hash(&system_id)),
        true,
        "process tree <-> System branch"
    );
    // make sure every process is in our tree and linked to its parent
    for (parent, child) in &chains {
        // get the ids for this parent and child process
        let parent_id = get_entity(&processes, parent)?;
        let child_id = get_entity(&processes, child)?;
        // make sure both processes are in our tree
        contains_key!(tree.data_map, &entity_hash(&parent_id));
        contains_key!(tree.data_map, &entity_hash(&child_id));
        // make sure this parent and child are linked
        is!(
            has_branch(&tree, entity_hash(&parent_id), entity_hash(&child_id)),
            true,
            "parent <-> child process branch"
        );
    }
    Ok(())
}

/// Test building a tree through a carved filesystem's entities
///
/// This guards the filesystem entity chain where a sample is linked to a
/// filesystem entity (`FileSystemIn`), folders are nested under it
/// (`FolderIn`), and files are linked into folders (`FileIn`). Starting a
/// tree from the parent firmware sample must walk the whole chain from
/// sample -> filesystem -> folders -> file samples, guarding against
/// entity to sample transitions breaking mid tree.
#[tokio::test]
async fn filesystem() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload a firmware sample to carve a filesystem from
    let firmware = client.files.create(gen_root(&group)).await?.sha256;
    // build an on disk filesystem to ingest
    let root_dir = std::env::temp_dir().join(format!("thorium-tree-tests-{}", Uuid::new_v4()));
    // create the folders in our filesystem
    tokio::fs::create_dir_all(root_dir.join("etc")).await?;
    tokio::fs::create_dir_all(root_dir.join("bin")).await?;
    // build the paths to the files in our filesystem
    let passwd = root_dir.join("etc/passwd");
    let busybox = root_dir.join("bin/busybox");
    // write random data to our files so reruns don't collide
    tokio::fs::write(&passwd, random_buffer()).await?;
    tokio::fs::write(&busybox, random_buffer()).await?;
    // the groups to build this filesystem in
    let groups = vec![group.clone()];
    // upload our filesystem's files as samples so their tree nodes resolve
    let passwd_sha = client
        .files
        .create(SampleRequest::new(&passwd, groups.clone()))
        .await?
        .sha256;
    let busybox_sha = client
        .files
        .create(SampleRequest::new(&busybox, groups.clone()))
        .await?
        .sha256;
    // build our filesystem entity
    let mut fs_builder = FileSystemEntityBuilder::new("tree-tests-fs", &root_dir)?;
    // add our files to this filesystem
    fs_builder.file(&passwd)?;
    fs_builder.file(&busybox)?;
    // ingest this filesystem into Thorium
    fs_builder
        .create(&"tree-tests".to_owned(), &firmware, &groups, &client)
        .await?;
    // clean up our on disk filesystem
    tokio::fs::remove_dir_all(&root_dir).await?;
    // start a tree from our firmware sample
    let tree = client
        .trees
        .start(&TreeOpts::default(), &start_query(&group, &firmware))
        .await?;
    // resolve our filesystem and folder entities to their ids
    let filesystems = entity_map(&client, &group, EntityKinds::FileSystem).await?;
    let folders = entity_map(&client, &group, EntityKinds::Folder).await?;
    // get the ids for our filesystem and its folders
    let fs_id = get_entity(&filesystems, "tree-tests-fs")?;
    let root_folder_id = get_entity(&folders, "/")?;
    let etc_id = get_entity(&folders, "etc")?;
    let bin_id = get_entity(&folders, "bin")?;
    // make sure our tree contains our firmware, filesystem, 3 folders, and 2 files
    is!(tree.data_map.len(), 7);
    contains_key!(tree.data_map, &sample_hash(&firmware));
    contains_key!(tree.data_map, &entity_hash(&fs_id));
    contains_key!(tree.data_map, &entity_hash(&root_folder_id));
    contains_key!(tree.data_map, &entity_hash(&etc_id));
    contains_key!(tree.data_map, &entity_hash(&bin_id));
    contains_key!(tree.data_map, &sample_hash(&passwd_sha));
    contains_key!(tree.data_map, &sample_hash(&busybox_sha));
    // make sure our filesystem is linked to our firmware
    is!(
        has_branch(&tree, sample_hash(&firmware), entity_hash(&fs_id)),
        true,
        "firmware <-> filesystem branch"
    );
    // make sure our folders are linked to our filesystem
    is!(
        has_branch(&tree, entity_hash(&fs_id), entity_hash(&root_folder_id)),
        true,
        "filesystem <-> root folder branch"
    );
    is!(
        has_branch(&tree, entity_hash(&root_folder_id), entity_hash(&etc_id)),
        true,
        "root folder <-> etc branch"
    );
    is!(
        has_branch(&tree, entity_hash(&root_folder_id), entity_hash(&bin_id)),
        true,
        "root folder <-> bin branch"
    );
    // make sure our files are linked to their folders
    is!(
        has_branch(&tree, entity_hash(&etc_id), sample_hash(&passwd_sha)),
        true,
        "etc <-> passwd branch"
    );
    is!(
        has_branch(&tree, entity_hash(&bin_id), sample_hash(&busybox_sha)),
        true,
        "bin <-> busybox branch"
    );
    Ok(())
}

/// Test building a tree with flags on samples and other entities
///
/// This guards nodes that have multiple associations of different kinds at
/// once: the root sample has both a `ProcessTreeIn` and a `FlagFor`
/// association, the child sample has a flag, and a process entity has a
/// flag. Every flag and the process tree must all appear alongside the
/// origin branches. This is a regression test for a bug where only the
/// first association per node (and only the first node with associations
/// per ring) was added to the tree because `scc`'s `iter_async` stops
/// iterating when its closure returns false.
#[tokio::test]
async fn flags() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload a root sample
    let root = client.files.create(gen_root(&group)).await?.sha256;
    // upload a single child unpacked from our root
    let child = upload_children(&client, &group, &root, 1).await?.remove(0);
    // build a tiny process tree to flag a process entity in
    let mut builder =
        WindowsProcessTreeEntity::builder("tree-tests-flagged-tree").tool("volatility");
    builder.add_mut(
        WindowsProcessEntity::new(1234)
            .name("evil.exe")
            .image_path(r"C:\Users\corn\evil.exe")
            .command("evil.exe -p 1337"),
    );
    // the groups to build this process tree in
    let groups = vec![group.clone()];
    // create our process tree entities
    let tree_id = builder.create_all(&groups, &client).await?;
    // our root sample is the source of the process tree association
    let source = AssociationTarget::File(root.clone());
    // our process tree entity is the target of the process tree association
    let target = AssociationTarget::Entity {
        id: tree_id,
        name: "tree-tests-flagged-tree".to_owned(),
    };
    // build the association linking our process tree to our root sample
    let assoc_req = AssociationRequest::new(AssociationKind::ProcessTreeIn, source)
        .target(target)
        .groups(groups.clone());
    // create this association
    client.associations.create(&assoc_req).await?;
    // resolve our process entity to its id
    let processes = entity_map(&client, &group, EntityKinds::WindowsProcess).await?;
    let evil_id = get_entity(&processes, "evil.exe")?;
    // flag our root sample
    let root_flag = flag_entity(
        &client,
        &group,
        "high-entropy",
        "This sample has a high entropy section",
        AssociationTarget::File(root.clone()),
    )
    .await?;
    // flag our child sample
    let child_flag = flag_entity(
        &client,
        &group,
        "known-bad-config",
        "This sample matches a known bad config",
        AssociationTarget::File(child.clone()),
    )
    .await?;
    // flag our process entity
    let proc_flag = flag_entity(
        &client,
        &group,
        "suspicious-process",
        "This process spawned from a user directory",
        AssociationTarget::Entity {
            id: evil_id,
            name: "evil.exe".to_owned(),
        },
    )
    .await?;
    // start a tree from our root sample
    let tree = client
        .trees
        .start(&TreeOpts::default(), &start_query(&group, &root))
        .await?;
    // make sure our tree contains our samples, process tree, process, and all 3 flags
    contains_key!(tree.data_map, &sample_hash(&root));
    contains_key!(tree.data_map, &sample_hash(&child));
    contains_key!(tree.data_map, &entity_hash(&tree_id));
    contains_key!(tree.data_map, &entity_hash(&evil_id));
    contains_key!(tree.data_map, &entity_hash(&root_flag));
    contains_key!(tree.data_map, &entity_hash(&child_flag));
    contains_key!(tree.data_map, &entity_hash(&proc_flag));
    is!(tree.data_map.len(), 7);
    // make sure each flag is linked to the data it flagged
    is!(
        has_branch(&tree, entity_hash(&root_flag), sample_hash(&root)),
        true,
        "flag <-> root sample branch"
    );
    is!(
        has_branch(&tree, entity_hash(&child_flag), sample_hash(&child)),
        true,
        "flag <-> child sample branch"
    );
    is!(
        has_branch(&tree, entity_hash(&proc_flag), entity_hash(&evil_id)),
        true,
        "flag <-> process branch"
    );
    // make sure our origin and process tree branches are still present
    is!(
        has_branch(&tree, sample_hash(&root), sample_hash(&child)),
        true,
        "root <-> child branch"
    );
    is!(
        has_branch(&tree, sample_hash(&root), entity_hash(&tree_id)),
        true,
        "root <-> process tree branch"
    );
    is!(
        has_branch(&tree, entity_hash(&tree_id), entity_hash(&evil_id)),
        true,
        "process tree <-> process branch"
    );
    Ok(())
}

/// Test growing a tree from multiple frontier nodes at once
///
/// This guards the `PATCH /api/trees/:cursor` grow route when the grow
/// query contains more than one growable node. Starting a 2 level tree
/// with a limit of 1 must leave all 3 children in `growable` and a single
/// grow from all of them at once must gather all 9 grandchildren with
/// branches back to their parents, guarding against grows that only
/// expand the first requested node.
#[tokio::test]
async fn grow_wide() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload a root sample
    let root = client.files.create(gen_root(&group)).await?.sha256;
    // upload 3 children unpacked from our root
    let children = upload_children(&client, &group, &root, 3).await?;
    // upload 3 grandchildren under each child
    let mut grandchildren = Vec::with_capacity(9);
    for child in &children {
        // upload this child's children
        grandchildren.extend(upload_children(&client, &group, child, 3).await?);
    }
    // build the opts to only grow our tree 1 level at a time
    let opts = TreeOpts::default().limit(1);
    // start a tree from our root sample
    let tree = client
        .trees
        .start(&opts, &start_query(&group, &root))
        .await?;
    // make sure our tree only contains our root and its children
    is!(tree.data_map.len(), 4);
    // make sure all of our children are on the growable frontier
    is!(tree.growable.len(), 3);
    for sha256 in &children {
        contains!(tree.growable, &sample_hash(sha256));
    }
    // build a query to grow this tree from its entire frontier
    let mut grow_query = TreeGrowQuery::default();
    // grow from every growable node in our tree
    for hash in &tree.growable {
        grow_query.add_growable_ref(*hash);
    }
    // grow this tree 1 more level from all 3 children at once
    let grown = client.trees.grow(tree.id, &opts, &grow_query).await?;
    // make sure we grew the same tree
    is!(grown.id, tree.id);
    // make sure every grandchild was gathered and linked to its parent
    for (index, sha256) in grandchildren.iter().enumerate() {
        // get the hash for this grandchild
        let grandchild_hash = sample_hash(sha256);
        // get the hash for this grandchild's parent
        let parent_hash = sample_hash(&children[index / 3]);
        // make sure this grandchild exists and is connected to its parent
        contains_key!(grown.data_map, &grandchild_hash);
        is!(
            has_branch(&grown, parent_hash, grandchild_hash),
            true,
            "child <-> grandchild branch"
        );
    }
    // make sure only the newly grown nodes were returned
    is!(grown.data_map.len(), 9);
    Ok(())
}

/// Test growing a tree across entity association hops
///
/// This guards growing trees through entities instead of just sample
/// origins. Starting a tree with a limit of 1 from a memory dump sample
/// must stop at the process tree entity and leave it growable. Growing
/// from that entity must then walk its `ChildProcess` associations to
/// gather the process entities below it, guarding against the grow route
/// failing to expand entity nodes loaded from a saved tree.
#[tokio::test]
async fn grow_entities() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload a memory dump sample to hang our process tree on
    let dump = client.files.create(gen_root(&group)).await?.sha256;
    // build a small windows process tree to grow into
    let mut builder =
        WindowsProcessTreeEntity::builder("tree-tests-grow-tree").tool("volatility");
    builder.add_mut(WindowsProcessEntity::new(4).name("System").image_path("System"));
    builder.add_mut(
        WindowsProcessEntity::new(368)
            .parent_pid(4)
            .name("smss.exe")
            .image_path(r"\SystemRoot\System32\smss.exe"),
    );
    // the groups to build this process tree in
    let groups = vec![group.clone()];
    // create our process tree entities
    let tree_id = builder.create_all(&groups, &client).await?;
    // our memory dump is the source of the process tree association
    let source = AssociationTarget::File(dump.clone());
    // our process tree entity is the target of the process tree association
    let target = AssociationTarget::Entity {
        id: tree_id,
        name: "tree-tests-grow-tree".to_owned(),
    };
    // build the association linking our process tree to our memory dump
    let assoc_req = AssociationRequest::new(AssociationKind::ProcessTreeIn, source)
        .target(target)
        .groups(groups.clone());
    // create this association
    client.associations.create(&assoc_req).await?;
    // resolve our process entities to their ids
    let processes = entity_map(&client, &group, EntityKinds::WindowsProcess).await?;
    let system_id = get_entity(&processes, "System")?;
    let smss_id = get_entity(&processes, "smss.exe")?;
    // start a tree that stops at our process tree entity
    let tree = client
        .trees
        .start(&TreeOpts::default().limit(1), &start_query(&group, &dump))
        .await?;
    // make sure our tree only contains our dump and process tree entity
    is!(tree.data_map.len(), 2);
    contains_key!(tree.data_map, &sample_hash(&dump));
    contains_key!(tree.data_map, &entity_hash(&tree_id));
    // make sure our process tree entity is on the growable frontier
    contains!(tree.growable, &entity_hash(&tree_id));
    // build a query to grow this tree from our process tree entity
    let grow_query = TreeGrowQuery::default().add_growable(entity_hash(&tree_id));
    // grow this tree 2 more levels into our process entities
    let grown = client
        .trees
        .grow(tree.id, &TreeOpts::default().limit(2), &grow_query)
        .await?;
    // make sure we grew the same tree
    is!(grown.id, tree.id);
    // make sure our process entities were gathered
    contains_key!(grown.data_map, &entity_hash(&system_id));
    contains_key!(grown.data_map, &entity_hash(&smss_id));
    // make sure our already sent nodes were trimmed
    is!(
        grown.data_map.contains_key(&sample_hash(&dump)),
        false,
        "already sent nodes should be trimmed"
    );
    // make sure our process entities are linked to the process tree
    is!(
        has_branch(&grown, entity_hash(&tree_id), entity_hash(&system_id)),
        true,
        "process tree <-> System branch"
    );
    is!(
        has_branch(&grown, entity_hash(&system_id), entity_hash(&smss_id)),
        true,
        "System <-> smss branch"
    );
    Ok(())
}
