//! The entity for storing filesystem data in thorium

use data_encoding::HEXLOWER;
use gxhash::HashSet;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use uuid::Uuid;

#[cfg(feature = "client")]
use futures::stream::{self, StreamExt};

#[cfg(feature = "client")]
use crate::models::{
    AssociationKind, AssociationRequest, AssociationTarget, EntityListOpts, EntityMetadata,
    EntityMetadataRequest, EntityRequest,
};
#[cfg(feature = "client")]
use crate::utils::helpers;

/// A filesystem entity
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct FileSystemEntity {
    /// The sha256 for this filesystem
    pub sha256: String,
    /// The tool(s) that dropped this filesystem
    pub tools: Vec<String>,
}

impl FileSystemEntity {
    /// Create a new file system entity with the info in the form
    ///
    /// # Errors
    ///
    /// * The vendor in the form does not exist in the given groups
    /// * An error occurred checking if the vendor in the form exists
    ///
    /// # Arguments
    ///
    /// * `form` -  The update form
    #[cfg(feature = "api")]
    pub fn from_form(form: super::EntityMetadataForm) -> Result<Self, crate::utils::ApiError> {
        // if we don't have the sha256 field then return an error
        let sha256 = match form.sha256 {
            Some(sha256) => sha256,
            None => return crate::bad!("File system entities must have a sha256!".to_owned()),
        };
        // build our file system entity
        Ok(FileSystemEntity {
            sha256,
            tools: form.tools,
        })
    }

    /// Add this filesystem entity metadata to a form
    ///
    /// # Arguments
    ///
    /// * `form` - The form to add too
    #[cfg(feature = "client")]
    pub fn add_to_form(
        self,
        form: reqwest::multipart::Form,
    ) -> Result<reqwest::multipart::Form, crate::Error> {
        // always set our entity kind
        let form = form.text("kind", super::EntityKinds::FileSystem.as_str());
        // set the sha256 for this entity
        let mut form = form.text("metadata[sha256]", self.sha256);
        // add our device metadata
        for tool in self.tools {
            form = form.text("metadata[tools][]", tool);
        }
        Ok(form)
    }
}

/// A file system folder entity
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct FileSystemFolderEntity {
    /// The id of the filesystem this fodler is from
    pub filesystem_id: Uuid,
    /// The sha256 for just the file names in this folder
    pub names_sha256: String,
    /// The sha256 for just the file data/contents (not names) in this folder
    pub data_sha256: String,
    /// The sha256 for the both the file names and its contents in this folder
    pub all_sha256: String,
}

impl FileSystemFolderEntity {
    /// Create a new file system entity with the info in the form
    ///
    /// # Errors
    ///
    /// * The vendor in the form does not exist in the given groups
    /// * An error occurred checking if the vendor in the form exists
    ///
    /// # Arguments
    ///
    /// * `form` -  The update form
    #[cfg(feature = "api")]
    pub fn from_form(form: super::EntityMetadataForm) -> Result<Self, crate::utils::ApiError> {
        // if we don't have the sha256 field then return an error
        let filesystem_id = match form.filesystem_id {
            Some(names_sha256) => names_sha256,
            None => {
                return crate::bad!(
                    "File system folder entities must have a filesystem id!".to_owned()
                );
            }
        };
        // if we don't have the sha256 field then return an error
        let names_sha256 = match form.names_sha256 {
            Some(names_sha256) => names_sha256,
            None => {
                return crate::bad!(
                    "File system folder entities must have a names sha256!".to_owned()
                );
            }
        };
        // if we don't have the sha256 field then return an error
        let data_sha256 = match form.data_sha256 {
            Some(data_sha256) => data_sha256,
            None => {
                return crate::bad!(
                    "File system folder entities must have a data sha256!".to_owned()
                );
            }
        };
        // if we don't have the sha256 field then return an error
        let all_sha256 = match form.all_sha256 {
            Some(all_sha256) => all_sha256,
            None => {
                return crate::bad!(
                    "File system folder entities must have an all sha256!".to_owned()
                );
            }
        };
        // build our file system entity
        Ok(FileSystemFolderEntity {
            filesystem_id,
            names_sha256,
            data_sha256,
            all_sha256,
        })
    }

    /// Add this filesystem entity metadata to a form
    ///
    /// # Arguments
    ///
    /// * `form` - The form to add too
    #[cfg(feature = "client")]
    pub fn add_to_form(
        self,
        form: reqwest::multipart::Form,
    ) -> Result<reqwest::multipart::Form, crate::Error> {
        // always set our entity kind
        let form = form.text("kind", super::EntityKinds::Folder.as_str());
        // set the sha256 for this entity
        let form = form.text("metadata[filesystem_id]", self.filesystem_id.to_string());
        let form = form.text("metadata[names_sha256]", self.names_sha256);
        let form = form.text("metadata[data_sha256]", self.data_sha256);
        let form = form.text("metadata[all_sha256]", self.all_sha256);
        Ok(form)
    }
}

struct FileSystemEntityBuilderContext {
    /// The id for this filesystem
    id: Uuid,
    /// The name for this filesystem
    name: String,
    /// A map of folders and their entity ids
    folder_ids: HashMap<PathBuf, (Uuid, String)>,
    /// Whether this was a new or existing filesystem
    already_exists: bool,
}

impl FileSystemEntityBuilderContext {
    /// Get the parent id for a path
    ///
    /// # Arguments
    ///
    /// * `path` - The path to get a parent id for
    pub fn get_parent(&self, path: &Path) -> Result<(Uuid, String), crate::Error> {
        // get our parent path or our filesystem id if we don't have a root
        let parent = match path.parent() {
            Some(parent) => parent.to_path_buf(),
            None => return Ok((self.id, self.name.clone())),
        };
        // get our parent paths id
        match self.folder_ids.get(&parent) {
            Some((id, name)) => Ok((*id, name.clone())),
            None => Err(crate::Error::new(format!(
                "Missing parent id for {}",
                path.display()
            ))),
        }
    }

    /// Add a folders id
    ///
    /// # Arguments
    ///
    /// * `path` - The path to this folder
    /// * `name` - The name of this folder
    /// * `id` - This folders entity id
    pub fn add_folder(&mut self, path: impl Into<PathBuf>, name: impl Into<String>, id: Uuid) {
        self.folder_ids.insert(path.into(), (id, name.into()));
    }
}

/// Constructs a filesystem that a user wants to submit to Thorium
#[derive(Debug)]
pub struct FileSystemEntityBuilder {
    /// The name of this filesystem
    pub name: String,
    /// The description for this filesystem
    pub description: Option<String>,
    /// Any tags to add to this filesystem
    pub tags: HashMap<String, HashSet<String>>,
    /// The paths to all files in this filesystem
    pub files: Vec<PathBuf>,
    /// The stripped path to the files or folders in this filesystem (folders will have the optional path set to None)
    pub contents: BTreeMap<PathBuf, BTreeMap<String, Option<PathBuf>>>,
    /// A map of paths and their sha256s
    sha256s: BTreeMap<PathBuf, String>,
    /// The root of this filesystem
    pub root: PathBuf,
}

impl FileSystemEntityBuilder {
    /// Create a new filesystem entity builder
    ///
    /// # Arguments
    ///
    /// * `name` - The name to use for this filesystem
    #[cfg(feature = "client")]
    pub fn new(name: impl Into<String>, root: impl Into<PathBuf>) -> Result<Self, crate::Error> {
        // get root path as a path
        let root = root.into();
        // init a default empty builder
        let mut builder = FileSystemEntityBuilder {
            name: name.into(),
            description: None,
            tags: HashMap::default(),
            files: Vec::with_capacity(100),
            contents: BTreeMap::default(),
            sha256s: BTreeMap::default(),
            root: root.clone(),
        };
        // add a root directory
        let entry = builder.contents.entry(PathBuf::from("/")).or_default();
        //// add a None entry for this folder since it won't have a data hash
        //entry.insert("/".to_owned(), None);
        Ok(builder)
    }

    /// Set the description for this filesystem
    ///
    /// # Arguments
    ///
    /// * `description` - The description to set
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        // update our description
        self.description = Some(description.into());
        self
    }

    /// Add a tag to this filesystem builder
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the tag to add
    /// * `value` - The value for the tag to add
    pub fn tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        // get an entry into this tags key list
        let entry = self.tags.entry(key.into()).or_default();
        // add our value
        entry.insert(value.into());
        self
    }

    /// Add a tag to this filesystem builder by mutable reference
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the tag to add
    /// * `value` - The value for the tag to add
    pub fn tag_mut(&mut self, key: impl Into<String>, value: impl Into<String>) {
        // get an entry into this tags key list
        let entry = self.tags.entry(key.into()).or_default();
        // add our value
        entry.insert(value.into());
    }

    /// Check if this filesystem contains data or not
    #[must_use]
    pub fn is_empty(&self) -> bool {
        // if we only have one folder then check if its just our root folder
        match self.contents.len() {
            0 => true,
            1 => {
                // if a builder only contains a single folder then its the root folder
                // if its empty then this fs is empty
                self.contents.iter().all(|(_, inner)| inner.is_empty())
            }
            _ => false,
        }
    }

    /// Add a folder to this filesystem
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the folder to add
    #[cfg(feature = "client")]
    pub fn directory<P: AsRef<Path>>(&mut self, path: P) -> Result<(), crate::client::Error> {
        // get our path from a ref
        let path = path.as_ref();
        // trim the prefix from our path
        let stripped = Path::new("/").join(path.strip_prefix(&self.root)?);
        // get our folders name as a string
        let name = match stripped.file_name() {
            Some(name) => name.to_string_lossy().to_string(),
            None => {
                return Err(crate::client::Error::new(format!(
                    "{} has no filename?",
                    path.display()
                )));
            }
        };
        // get an entry to this dirs
        let entry = self.contents.entry(stripped).or_default();
        // add a None entry for this folder since it won't have a data hash
        entry.insert(name, None);
        Ok(())
    }

    /// Add a file to this filesystem
    #[cfg(feature = "client")]
    pub fn file(&mut self, path: impl Into<PathBuf>) -> Result<(), crate::client::Error> {
        // convert our path into a path
        let path = path.into();
        // trim the prefix from our path
        let stripped = Path::new("/").join(path.strip_prefix(&self.root)?);
        // make sure our parent path exists
        let entry = match stripped.parent() {
            Some(parent) => self.contents.entry(parent.to_path_buf()).or_default(),
            // this should be unreachable since we prefix paths with '/'
            None => self.contents.entry(PathBuf::from("/")).or_default(),
        };
        // get our files name as a string
        let name = match stripped.file_name() {
            Some(name) => name.to_string_lossy().to_string(),
            None => {
                return Err(crate::client::Error::new(format!(
                    "{} has no filename?",
                    path.display()
                )));
            }
        };
        // add this file and it sha256
        entry.insert(name, Some(path.clone()));
        // add this path to our list of files
        self.files.push(path);
        Ok(())
    }

    /// Add a files sha256 to this filesystem
    pub fn add_sha256(&mut self, path: PathBuf, sha256: String) {
        self.sha256s.insert(path, sha256);
    }

    /// Remove a file from this filesystem
    ///
    /// # Arguments
    ///
    /// * `path` - The full absolute path of the file to remove
    #[cfg(feature = "client")]
    pub fn remove(&mut self, path: &PathBuf) -> Result<(), crate::client::Error> {
        // trim the prefix from our path
        let stripped = Path::new("/").join(path.strip_prefix(&self.root)?);
        // make sure our parent path exists
        let entry = match stripped.parent() {
            Some(parent) => self.contents.entry(parent.to_path_buf()).or_default(),
            // this should be unreachable since we prefix paths with '/'
            None => self.contents.entry(PathBuf::from("/")).or_default(),
        };
        // get our files name as a string
        let name = match path.file_name() {
            Some(name) => name.to_string_lossy().to_string(),
            None => {
                return Err(crate::client::Error::new(format!(
                    "{} has no filename?",
                    path.display()
                )));
            }
        };
        // remove this file
        entry.remove(&name);
        // remove from the file list as well
        self.files.retain(|item| item != path);
        Ok(())
    }

    /// Remove all empty folders
    pub fn clear_empty(&mut self) {
        // build a set of all paths that do contain files
        let has_files = self
            .contents
            .iter()
            .filter(|(_, folder)| !folder.is_empty())
            .map(|(path, _)| path)
            .collect::<Vec<&PathBuf>>();
        // build a set of all paths that do not contain files
        let empty_paths_iter = self
            .contents
            .iter()
            .filter(|(_, folder)| folder.is_empty())
            .map(|(path, _)| path)
            // check to see if any of these paths are subpaths for other folders
            .filter(|empty_path| {
                !has_files
                    .iter()
                    .rev()
                    .any(|has_path| has_path.starts_with(empty_path))
            })
            .map(ToOwned::to_owned);
        // build a hashset from our empty paths
        let empty_paths = empty_paths_iter.collect::<HashSet<_>>();
        // remove these paths from our contents
        self.contents.retain(|path, _| !empty_paths.contains(path));
    }

    /// Get a files sha256 by path
    ///
    /// # Arguments
    ///
    /// * `path` - The path to get a sha256 for
    #[cfg(feature = "client")]
    pub async fn get_hash_for_file(&self, path: &PathBuf) -> Result<String, crate::client::Error> {
        // check if this is already in our sha256 map
        match self.sha256s.get(path) {
            Some(sha256) => Ok(sha256.to_owned()),
            // we don't have this files sha256 so hash it manually
            None => helpers::sha256_file(path).await,
        }
    }

    /// Get the this filesystems sha256
    #[cfg(feature = "client")]
    pub async fn get_hash(&self) -> Result<String, crate::client::Error> {
        // hash the filesystem we want to add
        let mut hasher = Sha256::new();
        // hash all of the content in this filesystem
        for (path, files) in &self.contents {
            // hash this directories path
            hasher.update(path.as_os_str().as_encoded_bytes());
            // hash the files in this directory
            for (name, path) in files {
                // hash this files name
                hasher.update(name.as_bytes());
                // folders don't have data hashes
                if let Some(path) = path {
                    // add this files sha256 to our filesystem hash
                    match self.sha256s.get(path) {
                        Some(sha256) => hasher.update(sha256.as_bytes()),
                        None => {
                            // get this files sha256
                            let sha256 = helpers::sha256_file(path).await?;
                            // insert this sha256 into our sha256 map
                            hasher.update(sha256.as_bytes());
                        }
                    }
                }
            }
        }
        // get this filesystems hash
        Ok(HEXLOWER.encode(&hasher.finalize()))
    }

    /// Create the root filesystem entity
    ///
    /// # Arguments
    ///
    /// * `tool` - The tool that dumped/carved/created this filesystem
    /// * `parent_sha256` - The sha256 of this file this filesystem came from
    /// * `groups` - The groups to add this filesystem too
    /// * `thorium` - A Thorium client
    #[cfg(feature = "client")]
    async fn create_root(
        &self,
        tool: &String,
        parent_sha256: &str,
        groups: &[String],
        thorium: &crate::Thorium,
    ) -> Result<FileSystemEntityBuilderContext, crate::Error> {
        // get this filesystems hash
        let sha256 = self.get_hash().await?;
        // build the opts to get our parent hashes entities
        let opts = EntityListOpts::default().tag("Parent", parent_sha256);
        // get any filesystems with our fs hash
        let mut cursor = thorium.entities.list_details(&opts).await?;
        // find if this filesystem already exists
        loop {
            // check all of these entities for matching filesystems
            for entity in &cursor.data {
                // skip any entity that aren't filesystem
                if let EntityMetadata::FileSystem(meta) = &entity.metadata {
                    // if our sha256 matches then just use that filesystem
                    if meta.sha256 == sha256 {
                        // add our tool if this filesystem doesn't contain our tool name
                        if !meta.tools.contains(tool) {
                            // build an update to add this tool to this entity
                            let update = crate::models::EntityUpdate::default().tool(tool);
                            // update this entity
                            thorium.entities.update(entity.id, update).await?;
                        }
                        //build a context for this new filesystem
                        let context = FileSystemEntityBuilderContext {
                            id: entity.id,
                            name: entity.name.clone(),
                            folder_ids: HashMap::with_capacity(self.contents.len()),
                            already_exists: true,
                        };
                        // we don't need to make a new filesystem or assocation
                        return Ok(context);
                    }
                }
            }
            // if this cursor is exhausted then we didn't find an existing filesystem
            if cursor.exhausted() {
                // break and create a new fs to add
                break;
            }
            // get the next page of data
            cursor.refill().await?;
        }
        // create our root filesystem entity metadata
        let metadata = EntityMetadataRequest::FileSystem(FileSystemEntity {
            sha256: sha256.clone(),
            tools: Vec::default(),
        });
        // create an entity request for this filesystem
        let mut fs_req =
            EntityRequest::new(&self.name, metadata, groups).tag("Parent", parent_sha256);
        // add the rest of our builders tags
        for (key, values) in &self.tags {
            // get an entry to this tag keys values
            let entry = fs_req.tags.entry(key.clone()).or_default();
            // add this keys values
            entry.extend(values.iter().map(ToOwned::to_owned));
        }
        // create a filesystem entity
        let resp = thorium.entities.create(fs_req).await?;
        // build the source of this association
        let source = AssociationTarget::File(parent_sha256.to_owned());
        // build the target of this association
        let target = AssociationTarget::Entity {
            id: resp.id,
            name: self.name.clone(),
        };
        // build a base association request
        let assoc_req =
            AssociationRequest::new(AssociationKind::FileSystemIn, source).target(target);
        // create this association
        thorium.associations.create(&assoc_req).await?;
        //build a context for this new filesystem
        let context = FileSystemEntityBuilderContext {
            id: resp.id,
            name: self.name.clone(),
            folder_ids: HashMap::with_capacity(self.contents.len()),
            already_exists: false,
        };
        Ok(context)
    }

    /// Create and link a folder entity
    ///
    /// # Arguments
    ///
    /// * `path` - The path to this folder
    /// * `parent_stack` - The stack of parent entity ids and names to add this root too
    /// * `thorium` - A Thorium client
    #[cfg(feature = "client")]
    async fn create_folder(
        &self,
        path: &PathBuf,
        files: &BTreeMap<String, Option<PathBuf>>,
        groups: &[String],
        context: &mut FileSystemEntityBuilderContext,
        thorium: &crate::Thorium,
    ) -> Result<(Uuid, String), crate::Error> {
        // get our parents id
        let (parent_id, parent_name) = context.get_parent(path)?;
        // get this folders name as a string
        let folder_name = match path.file_name() {
            Some(name) => name.to_string_lossy().to_string(),
            None => "/".to_string(),
        };
        // generate a hash from just names
        let names_sha256 = helpers::sha256_iter(files.keys());
        // generate a hash from data
        let data_sha256 = helpers::sha256_iter(self.sha256s.values());
        // generate a hash from both names and data
        let all_sha256 = helpers::sha256_iter(files.keys().chain(self.sha256s.values()));
        // create a filesystem folder entity request
        let metadata = EntityMetadataRequest::Folder(FileSystemFolderEntity {
            filesystem_id: context.id,
            names_sha256,
            data_sha256,
            all_sha256,
        });
        // build an entity request for this folder
        let folder_req = EntityRequest::new(folder_name.clone(), metadata, groups);
        // create a folder entity
        let resp = thorium.entities.create(folder_req).await?;
        // build the target of this association
        let source = AssociationTarget::Entity {
            id: parent_id,
            name: parent_name.clone(),
        };
        // build the target of this association
        let target = AssociationTarget::Entity {
            id: resp.id,
            name: folder_name.clone(),
        };
        // build a base association request
        let assoc_req = AssociationRequest::new(AssociationKind::FolderIn, source).target(target);
        // create this association
        thorium.associations.create(&assoc_req).await?;
        // add our folders id onto our parent stack
        context.add_folder(path, &folder_name, resp.id);
        Ok((resp.id, folder_name))
    }

    /// Link a file to a folder in this filesystem
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the file to link
    /// * `folder_id` - The id of the folder to link this file to
    /// * `folder_name` - The name of the folder to link this file to
    /// * `thorium` - A Thorium client
    #[cfg(feature = "client")]
    async fn link_file(
        &self,
        path: &PathBuf,
        folder_id: Uuid,
        folder_name: String,
        thorium: &crate::Thorium,
    ) -> Result<(), crate::Error> {
        // get this files sha256
        let sha256 = self.get_hash_for_file(path).await?;
        // build the target of this association
        let source = AssociationTarget::Entity {
            id: folder_id,
            name: folder_name,
        };
        // build the target of this association
        let target = AssociationTarget::File(sha256);
        // build a base association request
        let assoc_req = AssociationRequest::new(AssociationKind::FileIn, source).target(target);
        // create this association
        thorium.associations.create(&assoc_req).await?;
        Ok(())
    }

    /// Create this filesystem structure through entities in Thorium
    ///
    /// # Panics
    ///
    /// idk man
    #[cfg(feature = "client")]
    pub async fn create(
        &self,
        tool: &String,
        parent_sha256: &str,
        groups: &[String],
        thorium: &crate::Thorium,
    ) -> Result<(), crate::Error> {
        // create our root fs entity and build a filesystem builder context
        let mut context = self
            .create_root(tool, parent_sha256, groups, thorium)
            .await?;
        // check if we have already ingested this filesytem
        if context.already_exists {
            // we already have a root fs so for now just assume the rest of the FS already exists
            return Ok(());
        }
        // start linking all files in this filesystem together
        for (path, files) in &self.contents {
            // create and link this folder
            let (folder_id, folder_name) = self
                .create_folder(path, files, groups, &mut context, thorium)
                .await?;
            // create associations for all of the files in this folder
            helpers::assert_send_stream(
                stream::iter(files.values().filter_map(|path| path.as_ref()))
                    .map(|path| self.link_file(path, folder_id, folder_name.clone(), thorium))
                    .buffer_unordered(10),
            )
            .collect::<Vec<Result<(), crate::Error>>>()
            .await;
        }
        Ok(())
    }
}
