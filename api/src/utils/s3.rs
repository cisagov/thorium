//! Handles uploading files to s3

use aws_credential_types::provider::SharedCredentialsProvider;
use aws_sdk_s3::primitives::SdkBody;
use aws_sdk_s3::types::{CompletedMultipartUpload, CompletedPart};
use aws_sdk_s3::{
    config::Credentials, operation::head_object::HeadObjectError, primitives::ByteStream, Client,
};
use axum::extract::multipart::Field;
use base64::Engine as _;
use bytes::{buf::Buf, BytesMut};
use cart_rs::{CartStreamManual, UncartStream};
use data_encoding::HEXLOWER;
use generic_array::{typenum::U16, GenericArray};
use md5::Md5;
use sha1::{Digest, Sha1};
use sha2::Sha256;
use std::io::Write;
use tracing::{event, instrument, Level};
use uuid::Uuid;
use zip::unstable::write::FileOptionsExt;
use zip::write::ZipWriter;

use super::{ApiError, Shared};
use crate::models::ZipDownloadParams;
use crate::{bad, unavailable, Conf};

/// A tuple of hashes (sha256, sha1, md5)
pub type Hashes = (String, String, String);

/// The standard hashes for a file
#[derive(Debug)]
pub struct StandardHashes {
    /// The sha256 hash
    pub sha256: String,
    /// The sha1 hash
    pub sha1: String,
    /// The md5 hash
    pub md5: String,
}

/// Hashes files with sha256, sha1, and md5
pub struct StandardHashers {
    /// The sha256 hasher
    pub sha256: Sha256,
    /// The sha1 hasher
    pub sha1: Sha1,
    /// The md5 hasher
    pub md5: Md5,
}

impl StandardHashers {
    /// Add a buffer to our hashers
    ///
    /// # Arguments
    ///
    /// * `buff` - The buffer to digest
    pub fn digest(&mut self, buff: &[u8]) {
        // digest this buffer with each of our hashers
        self.sha256.update(buff);
        self.sha1.update(buff);
        self.md5.update(buff);
    }

    /// Finalize our hashers and get our hashes
    pub fn finish(self) -> StandardHashes {
        // build our digests
        let sha256 = HEXLOWER.encode(&self.sha256.finalize());
        let sha1 = HEXLOWER.encode(&self.sha1.finalize());
        let md5 = HEXLOWER.encode(&self.md5.finalize());
        StandardHashes { sha256, sha1, md5 }
    }
}

impl Default for StandardHashers {
    /// Create default hashers
    fn default() -> Self {
        StandardHashers {
            sha256: Sha256::new(),
            sha1: Sha1::new(),
            md5: Md5::new(),
        }
    }
}

/// A S3 client wrapper
pub struct S3 {
    /// The s3 bucket for files
    pub files: S3Client,
    /// The s3 bucket for result files
    pub results: S3Client,
    /// The s3 bucket for ephemeral files
    pub ephemeral: S3Client,
    /// The s3 bucket for comment attachments
    pub attachments: S3Client,
    /// The s3 bucket for zipped repositories
    pub repos: S3Client,
}

impl S3 {
    /// Build all of our s3 clients
    pub fn new(config: &Conf) -> Self {
        // build our clients
        let files = S3Client::new(
            &config.thorium.files.bucket,
            &config.thorium.files.password,
            &config.thorium.s3,
        );
        let results = S3Client::new(
            &config.thorium.results.bucket,
            // these aren't password protected so just use the files password
            &config.thorium.files.password,
            &config.thorium.s3,
        );
        let ephemeral = S3Client::new(
            &config.thorium.ephemeral.bucket,
            // these aren't password protected so just use the files password
            &config.thorium.files.password,
            &config.thorium.s3,
        );
        let attachments = S3Client::new(
            &config.thorium.attachments.bucket,
            // these aren't password protected so just use the files password
            &config.thorium.files.password,
            &config.thorium.s3,
        );
        let repos = S3Client::new(
            &config.thorium.repos.bucket,
            // these aren't password protected so just use the files password
            &config.thorium.files.password,
            &config.thorium.s3,
        );
        S3 {
            files,
            results,
            ephemeral,
            attachments,
            repos,
        }
    }
}

pub struct S3Client {
    /// The bucket to write files too
    pub bucket: String,
    /// The password used to encrypt files
    password: GenericArray<u8, U16>,
    /// The test aws sdk s3 client
    pub client: Client,
}

impl S3Client {
    /// builds new s3 clients
    ///
    /// # Arguments
    ///
    /// * `config` - Thorium config options
    pub fn new(bucket: &str, password: &str, conf: &crate::conf::S3) -> Self {
        // build our generic array
        let gen_array: GenericArray<u8, U16> =
            GenericArray::clone_from_slice(&password.as_bytes()[..16]);
        // get our s3 credentials
        let creds = Credentials::new(&conf.access_key, &conf.secret_token, None, None, "Thorium");
        // build our s3 config
        let s3_config = aws_sdk_s3::config::Builder::new()
            .endpoint_url(&conf.endpoint)
            .region(aws_types::region::Region::new(conf.region.clone()))
            .credentials_provider(SharedCredentialsProvider::new(creds))
            .force_path_style(true)
            .build();
        // build our s3 client
        let client = Client::from_conf(s3_config);
        S3Client {
            bucket: bucket.to_owned(),
            password: gen_array,
            client,
        }
    }

    /// Check if a file exists in s3 by path
    ///
    /// # Arguments
    ///
    /// * `path` - The path to check against
    #[instrument(name = "S3Client::exists", skip(self), err(Debug))]
    pub async fn exists(&self, path: &str) -> Result<bool, ApiError> {
        // head this path to see if it exists
        match self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(path)
            .send()
            .await
        {
            Ok(_) => Ok(true),
            Err(sdk_err) => match sdk_err.into_service_error() {
                HeadObjectError::NotFound(_) => Ok(false),
                err => Err(ApiError::from(err)),
            },
        }
    }

    /// Stream a file into s3 while hashing and carting it
    ///
    /// # Arguments
    ///
    /// * `path` - The path to write this object to in s3
    /// * `upload_id` - The id of the multipart upload being used
    /// * `field` - The field to stream to s3
    #[instrument(
        name = "S3Client::hash_cart_and_stream_helper",
        skip(self, field),
        err(Debug)
    )]
    async fn hash_cart_and_stream_helper<'a>(
        &self,
        path: &str,
        upload_id: &str,
        mut field: Field<'a>,
    ) -> Result<StandardHashes, ApiError> {
        // init our cart streamer and hashers
        let mut cart = CartStreamManual::new(&self.password, 7_242_880)?;
        let mut hashers = StandardHashers::default();
        // track what part number we are on
        let mut part_num = 1;
        // keep a list of parts we have uploaded
        let mut parts = Vec::with_capacity(10);
        // stream this fields data through our hashers, cart, and to s3
        while let Some(raw) = field.chunk().await? {
            // pass this chunk through our hashers
            hashers.digest(&raw);
            // add this buffer to our cart streamer
            if cart.next_bytes(raw)? {
                // keep processing these bytes until they are finished
                while cart.process()? {
                    // if our input buffer is full then pack
                    if cart.ready() >= 5_242_880 {
                        // get the bytes we are ready to write to s3
                        let writable = cart.carted_bytes();
                        // pack our entire input buffer
                        let carted = ByteStream::from(SdkBody::from(writable));
                        // write this buffer to s3
                        let part = self
                            .client
                            .upload_part()
                            .bucket(&self.bucket)
                            .key(path)
                            .upload_id(upload_id)
                            .body(carted)
                            .part_number(part_num)
                            .send()
                            .await?;
                        // add this chunk to our parts list
                        parts.push(
                            CompletedPart::builder()
                                .e_tag(part.e_tag.unwrap_or_default())
                                .part_number(part_num)
                                .build(),
                        );
                        // consume the bytes we have written to s3
                        cart.consume();
                        // increment our part number
                        part_num += 1;
                    }
                }
            }
        }
        // finish carting our file
        let writable = cart.finish()?;
        // finish our carted file
        let carted = ByteStream::from(SdkBody::from(writable));
        // write this final buffer to s3
        let part = self
            .client
            .upload_part()
            .bucket(&self.bucket)
            .key(path)
            .upload_id(upload_id)
            .body(carted)
            .part_number(part_num)
            .send()
            .await?;
        // add this chunk to our parts list
        parts.push(
            CompletedPart::builder()
                .e_tag(part.e_tag.unwrap_or_default())
                .part_number(part_num)
                .build(),
        );
        // build our complete multipart upload object
        let completed_parts = CompletedMultipartUpload::builder()
            .set_parts(Some(parts))
            .build();
        // finish this multipart upload
        self.client
            .complete_multipart_upload()
            .bucket(&self.bucket)
            .key(path)
            .multipart_upload(completed_parts)
            .upload_id(upload_id)
            .send()
            .await?;
        Ok(hashers.finish())
    }

    /// Stream a file into s3 while hashing and carting it
    ///
    /// # Arguments
    ///
    /// * `s3_id` - The id to use for this object in s3
    /// * `field` - The field to stream to s3
    #[instrument(name = "S3Client::hash_cart_and_stream", skip(self, field), err(Debug))]
    pub async fn hash_cart_and_stream<'a>(
        &self,
        s3_id: &Uuid,
        field: Field<'a>,
    ) -> Result<StandardHashes, ApiError> {
        // build the path to write this file too
        let path = s3_id.to_string();
        // initiate a multipart upload to s3
        let init = self
            .client
            .create_multipart_upload()
            .bucket(&self.bucket)
            .key(&path)
            .content_type("application/octet-stream")
            .send()
            .await?;
        // get our upload id
        let upload_id = match init.upload_id() {
            Some(upload_id) => upload_id,
            None => return unavailable!("Failed to get multipart upload ID".to_owned()),
        };
        // cart and stream this file to s3
        match self
            .hash_cart_and_stream_helper(&path, upload_id, field)
            .await
        {
            Ok(hashes) => Ok(hashes),
            Err(err) => {
                // abort this multipart upload
                self.client
                    .abort_multipart_upload()
                    .bucket(&self.bucket)
                    .key(path)
                    .upload_id(upload_id)
                    .send()
                    .await?;
                // return our error
                return Err(ApiError::from(err));
            }
        }
    }

    /// Helps stream a file into s3 while sha256 and carting it
    ///
    /// # Arguments
    ///
    /// * `path` - The path to write this object to in s3
    /// * `upload_id` - The id of the multipart upload being used
    /// * `field` - The field to stream to s3
    #[instrument(
        name = "S3Client::sha256_cart_and_stream_helper",
        skip(self, field),
        err(Debug)
    )]
    async fn sha256_cart_and_stream_helper<'a>(
        &self,
        path: &str,
        upload_id: &str,
        mut field: Field<'a>,
    ) -> Result<String, ApiError> {
        // init our cart streamer and hashers
        let mut cart = CartStreamManual::new(&self.password, 7_242_880)?;
        let mut sha256 = Sha256::new();
        // track what part number we are on
        let mut part_num = 1;
        // keep a list of parts we have uploaded
        let mut parts = Vec::with_capacity(10);
        // stream this fields data through our hashers, cart, and to s3
        while let Some(raw) = field.chunk().await? {
            // pass this chunk through our hashers
            sha256.update(&raw);
            // add this buffer to our cart streamer
            if cart.next_bytes(raw)? {
                // keep processing these bytes until they are finished
                while cart.process()? {
                    // if our input buffer is full then pack
                    if cart.ready() >= 5_242_880 {
                        // get the bytes we are ready to write to s3
                        let writable = cart.carted_bytes();
                        // pack our entire input buffer
                        let carted = ByteStream::from(SdkBody::from(writable));
                        // write this buffer to s3
                        let part = self
                            .client
                            .upload_part()
                            .bucket(&self.bucket)
                            .key(path)
                            .upload_id(upload_id)
                            .body(carted)
                            .part_number(part_num)
                            .send()
                            .await?;
                        // add this chunk to our parts list
                        parts.push(
                            CompletedPart::builder()
                                .e_tag(part.e_tag.unwrap_or_default())
                                .part_number(part_num)
                                .build(),
                        );
                        // consume the bytes we have written to s3
                        cart.consume();
                        // increment our part number
                        part_num += 1;
                    }
                }
            }
        }
        // finish carting our file
        let writable = cart.finish()?;
        // finish our carted file
        let carted = ByteStream::from(SdkBody::from(writable));
        // write this final buffer to s3
        let part = self
            .client
            .upload_part()
            .bucket(&self.bucket)
            .key(path)
            .upload_id(upload_id)
            .body(carted)
            .part_number(part_num)
            .send()
            .await?;
        // add this chunk to our parts list
        parts.push(
            CompletedPart::builder()
                .e_tag(part.e_tag.unwrap_or_default())
                .part_number(part_num)
                .build(),
        );
        // build our complete multipart upload object
        let completed_parts = CompletedMultipartUpload::builder()
            .set_parts(Some(parts))
            .build();
        // finish this multipart upload
        self.client
            .complete_multipart_upload()
            .bucket(&self.bucket)
            .key(path)
            .multipart_upload(completed_parts)
            .upload_id(upload_id)
            .send()
            .await?;
        // get our final sha256 hash
        Ok(HEXLOWER.encode(&sha256.finalize()))
    }

    /// Stream a file into s3 while getting its sha256 and carting it
    ///
    /// # Arguments
    ///
    /// * `s3_id` - The id to use for this object in s3
    /// * `field` - The field to stream to s3
    #[instrument(
        name = "S3Client::sha256_cart_and_stream",
        skip(self, field),
        err(Debug)
    )]
    pub async fn sha256_cart_and_stream<'a>(
        &self,
        s3_id: &Uuid,
        field: Field<'a>,
    ) -> Result<String, ApiError> {
        // build the path to write this file too
        let path = s3_id.to_string();
        // initiate a multipart upload to s3
        let init = self
            .client
            .create_multipart_upload()
            .bucket(&self.bucket)
            .key(&path)
            .content_type("application/octet-stream")
            .send()
            .await?;
        // get our upload id
        let upload_id = match init.upload_id() {
            Some(upload_id) => upload_id,
            None => return unavailable!("Failed to get multipart upload ID".to_owned()),
        };
        // cart and stream this file to s3
        match self
            .sha256_cart_and_stream_helper(&path, upload_id, field)
            .await
        {
            Ok(sha256) => Ok(sha256),
            Err(err) => {
                // abort this multipart upload
                self.client
                    .abort_multipart_upload()
                    .bucket(&self.bucket)
                    .key(path)
                    .upload_id(upload_id)
                    .send()
                    .await?;
                // return our error
                return Err(ApiError::from(err));
            }
        }
    }

    /// Stream a file into s3 after carting it
    ///
    /// # Arguments
    ///
    /// * `path` - The path to write this object to in s3
    /// * `upload_id` - The id of the multipart upload being used
    /// * `field` - The field to stream to s3
    #[instrument(
        name = "S3Client::cart_and_stream_helper",
        skip(self, field),
        err(Debug)
    )]
    async fn cart_and_stream_helper<'a>(
        &self,
        path: &str,
        upload_id: &str,
        mut field: Field<'a>,
    ) -> Result<(), ApiError> {
        // init our cart streamer and hashers
        let mut cart = CartStreamManual::new(&self.password, 7_242_880)?;
        // track what part number we are on
        let mut part_num = 1;
        // keep a list of parts we have uploaded
        let mut parts = Vec::with_capacity(10);
        // stream this fields data through our hashers, cart, and to s3
        while let Some(raw) = field.chunk().await? {
            // add this buffer to our cart streamer
            if cart.next_bytes(raw)? {
                // keep processing these bytes until they are finished
                while cart.process()? {
                    // if our input buffer is full then pack
                    if cart.ready() >= 5_242_880 {
                        // get the bytes we are ready to write to s3
                        let writable = cart.carted_bytes();
                        // pack our entire input buffer
                        let carted = ByteStream::from(SdkBody::from(writable));
                        // write this buffer to s3
                        let part = self
                            .client
                            .upload_part()
                            .bucket(&self.bucket)
                            .key(path)
                            .upload_id(upload_id)
                            .body(carted)
                            .part_number(part_num)
                            .send()
                            .await?;
                        // add this chunk to our parts list
                        parts.push(
                            CompletedPart::builder()
                                .e_tag(part.e_tag.unwrap_or_default())
                                .part_number(part_num)
                                .build(),
                        );
                        // consume the bytes we have written to s3
                        cart.consume();
                        // increment our part number
                        part_num += 1;
                    }
                }
            }
        }
        // finish carting our file
        let writable = cart.finish()?;
        // finish our carted file
        let carted = ByteStream::from(SdkBody::from(writable));
        // write this final buffer to s3
        let part = self
            .client
            .upload_part()
            .bucket(&self.bucket)
            .key(path)
            .upload_id(upload_id)
            .body(carted)
            .part_number(part_num)
            .send()
            .await?;
        // add this chunk to our parts list
        parts.push(
            CompletedPart::builder()
                .e_tag(part.e_tag.unwrap_or_default())
                .part_number(part_num)
                .build(),
        );
        // build our complete multipart upload object
        let completed_parts = CompletedMultipartUpload::builder()
            .set_parts(Some(parts))
            .build();
        // finish this multipart upload
        self.client
            .complete_multipart_upload()
            .bucket(&self.bucket)
            .key(path)
            .multipart_upload(completed_parts)
            .upload_id(upload_id)
            .send()
            .await?;
        Ok(())
    }

    /// Stream a file into s3 after carting it
    ///
    /// # Arguments
    ///
    /// * `s3_id` - The id to use for this object in s3
    /// * `field` - The field to stream to s3
    #[instrument(name = "S3Client::cart_and_stream", skip(self, field), err(Debug))]
    pub async fn cart_and_stream<'a>(
        &self,
        s3_id: &Uuid,
        field: Field<'a>,
    ) -> Result<(), ApiError> {
        // build the path to write this file too
        let path = s3_id.to_string();
        // initiate a multipart upload to s3
        let init = self
            .client
            .create_multipart_upload()
            .bucket(&self.bucket)
            .key(&path)
            .content_type("application/octet-stream")
            .send()
            .await?;
        // get our upload id
        let upload_id = match init.upload_id() {
            Some(upload_id) => upload_id,
            None => return unavailable!("Failed to get multipart upload ID".to_owned()),
        };
        // cart and stream this file to s3
        match self.cart_and_stream_helper(&path, upload_id, field).await {
            Ok(()) => Ok(()),
            Err(err) => {
                // abort this multipart upload
                self.client
                    .abort_multipart_upload()
                    .bucket(&self.bucket)
                    .key(path)
                    .upload_id(upload_id)
                    .send()
                    .await?;
                // return our error
                return Err(ApiError::from(err));
            }
        }
    }

    /// Stream a file into s3 without carting it
    ///
    /// # Arguments
    ///
    /// * `path` - The path to write this object to in s3
    /// * `upload_id` - The id of the multipart upload being used
    /// * `field` - The field to stream to s3
    #[instrument(name = "S3Client::stream_helper", skip(self, field), err(Debug))]
    pub async fn stream_helper<'a>(
        &self,
        path: &str,
        upload_id: &str,
        mut field: Field<'a>,
    ) -> Result<(), ApiError> {
        // track what part number we are on
        let mut part_num = 1;
        // keep a list of parts we have uploaded
        let mut parts = Vec::with_capacity(1);
        // build our buffer so we can have at least 5 mebibytes of chunks to send
        let mut stream = BytesMut::with_capacity(7_242_880);
        // stream this fields data through our hashers, cart, and to s3
        while let Some(raw) = field.chunk().await? {
            // add our chunk to our stream buffer
            stream.extend_from_slice(&raw);
            // add this buffer to our cart streamer
            if stream.remaining() >= 5_242_880 {
                // pack our entire input buffer
                let carted = ByteStream::from(SdkBody::from(&stream[..]));
                // write this buffer to s3
                let part = self
                    .client
                    .upload_part()
                    .bucket(&self.bucket)
                    .key(path)
                    .upload_id(upload_id)
                    .body(carted)
                    .part_number(part_num)
                    .send()
                    .await?;
                // add this chunk to our parts list
                parts.push(
                    CompletedPart::builder()
                        .e_tag(part.e_tag.unwrap_or_default())
                        .part_number(part_num)
                        .build(),
                );
                // reset our packable and writable number of bytes to 0
                stream.clear();
                // increment our part number
                part_num += 1;
            }
        }
        // finish our stream
        let carted = ByteStream::from(SdkBody::from(&stream[..]));
        // write this buffer to s3
        let part = self
            .client
            .upload_part()
            .bucket(&self.bucket)
            .key(path)
            .upload_id(upload_id)
            .body(carted)
            .part_number(part_num)
            .send()
            .await?;
        // add this chunk to our parts list
        parts.push(
            CompletedPart::builder()
                .e_tag(part.e_tag.unwrap_or_default())
                .part_number(part_num)
                .build(),
        );
        // build our complete multipart upload object
        let completed_parts = CompletedMultipartUpload::builder()
            .set_parts(Some(parts))
            .build();
        // finish this multipart upload
        self.client
            .complete_multipart_upload()
            .bucket(&self.bucket)
            .key(path)
            .multipart_upload(completed_parts)
            .upload_id(upload_id)
            .send()
            .await?;
        Ok(())
    }

    /// Stream a file into s3
    ///
    /// # Arguments
    ///
    /// * `s3_id` - The id to use for this object in s3
    /// * `field` - The field to stream to s3
    #[instrument(name = "S3Client::stream", skip(self, field), err(Debug))]
    pub async fn stream<'a>(&self, path: &str, field: Field<'a>) -> Result<(), ApiError> {
        // ban any paths that might contain traversal attacks
        if path.contains("..") {
            return bad!("S3 file names cannot contain '..'".to_owned());
        }
        // initiate a multipart upload to s3
        let init = self
            .client
            .create_multipart_upload()
            .bucket(&self.bucket)
            .key(path)
            .content_type("application/octet-stream")
            .send()
            .await?;
        // get our upload id
        let upload_id = match init.upload_id() {
            Some(upload_id) => upload_id,
            None => return unavailable!("Failed to get multipart upload ID".to_owned()),
        };
        // cart and stream this file to s3
        match self.stream_helper(&path, upload_id, field).await {
            Ok(()) => Ok(()),
            Err(err) => {
                // abort this multipart upload
                self.client
                    .abort_multipart_upload()
                    .bucket(&self.bucket)
                    .key(path)
                    .upload_id(upload_id)
                    .send()
                    .await?;
                // return our error
                return Err(ApiError::from(err));
            }
        }
    }

    /// decodes a base64 stream and uploads it to s3
    ///
    /// # Arguments
    ///
    /// * `path` - The path to upload this file to
    /// * `encoded` - The base64 encoded data to upload
    #[instrument(name = "S3Client::upload_base_64", skip(self, encoded), err(Debug))]
    pub async fn upload_base64(&self, path: &str, encoded: &str) -> Result<(), ApiError> {
        // log the size of our encoded data
        event!(Level::INFO, encoded_size = encoded.len());
        // ban any paths that might contain traversal attacks
        if path.contains("..") {
            return bad!("S3 file names cannot contain '..'".to_owned());
        }
        // decode our base64'd bytes
        let decoded = base64::engine::general_purpose::STANDARD.decode(encoded)?;
        // decode this file
        let decoded_stream = ByteStream::from(decoded);
        // write this file to s3
        if !self.exists(path).await? {
            self.client
                .put_object()
                .bucket(&self.bucket)
                .key(path)
                .body(decoded_stream)
                .send()
                .await?;
        }
        Ok(())
    }

    /// download a file from s3
    ///
    /// # Arguments
    ///
    /// * `path` - The path to an object in s3
    #[instrument(name = "S3Client::download", skip(self), err(Debug))]
    pub async fn download(&self, path: &str) -> Result<ByteStream, ApiError> {
        // start downloading this file and stream it to the user
        let body = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(path)
            .send()
            .await?
            .body;
        Ok(body)
    }

    /// download a file from s3 and convert it to an encrypted zip
    ///
    /// This is not near as efficient as using CaRT and should not be used for large files.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to an object in s3
    #[instrument(name = "S3Client::download_as_zip", skip(self, shared), err(Debug))]
    pub async fn download_as_zip(
        &self,
        path: &str,
        sha256: &str,
        params: ZipDownloadParams,
        shared: &Shared,
    ) -> Result<Vec<u8>, ApiError> {
        // start downloading this file and stream it to the user
        let body = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(path)
            .send()
            .await?
            .body;
        // get the password to use
        let password = params.get_password(shared).as_bytes();
        // setup our zip options
        let opts = zip::write::SimpleFileOptions::default().with_deprecated_encryption(password);
        // build our writer
        let mut writer = ZipWriter::new(std::io::Cursor::new(vec![]));
        // start our file
        writer.start_file(sha256, opts)?;
        // build our uncart stream object
        let mut uncart_stream = UncartStream::new(body.into_async_read());
        // build a vector to store our entire file that defaults to 1 mebibyte in size
        let mut uncarted = Vec::with_capacity(1_048_576);
        // uncart the entire file
        tokio::io::copy(&mut uncart_stream, &mut uncarted).await?;
        // spawn this task in a tokio task and wait for it to complete
        tokio::task::spawn_blocking(move || {
            // zip this file
            match writer.write_all(&uncarted) {
                // get our zipped data
                Ok(_) => match writer.finish() {
                    Ok(zipped) => Ok(zipped.into_inner()),
                    Err(err) => Err(ApiError::from(err)),
                },
                Err(err) => Err(ApiError::from(err)),
            }
        })
        .await?
    }

    /// deletes a file from s3
    ///
    /// # Arguments
    ///
    /// * `path` - The path of the file to delete
    #[instrument(name = "S3Client::delete", skip(self), err(Debug))]
    pub async fn delete(&self, path: &str) -> Result<(), ApiError> {
        // try to delete this object from s3
        self.client
            .delete_object()
            .bucket(&self.bucket)
            .key(path)
            .send()
            .await?;
        Ok(())
    }
}
