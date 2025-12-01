use aws_credential_types::provider::SharedCredentialsProvider;
use aws_sdk_s3::{
    Client,
    config::Credentials,
    types::{BucketLocationConstraint, CreateBucketConfiguration},
};
use thorium::{Error, conf::S3};

use aws_sdk_s3::error::SdkError;
use aws_sdk_s3::operation::create_bucket::CreateBucketError;

use crate::k8s::clusters::ClusterMeta;

/// Build an API url string
///
/// Get the thorium host from operator args or the target ThoriumCluster instance being configured.
/// The url string will be none outside of a development environment when running in kubernetes
/// within a pod.
///
/// # Arguments
///
/// * `meta` - Thorium cluster client and metadata
/// * `url` - The Thorium API URL passed to the operator as an argument
pub fn get_thorium_host(meta: &ClusterMeta, url: Option<&String>) -> String {
    match url {
        // grab url if passed to the operator as an arg, mostly for development
        Some(url) => url.to_owned(),
        // use internal k8s networking by default
        None => {
            format!(
                "http://thorium-api.{}.svc.cluster.local:80",
                &meta.namespace
            )
        }
    }
}

/// Create an S3 bucket
///
/// # Arguments
///
/// * `config` - The Thorium S3 configuration
/// * `client` - API client for S3 interface
/// * `bucket_name` - Name of bucket to create
pub async fn create_bucket(config: &S3, client: &Client, bucket_name: &str) -> Result<(), Error> {
    // if bucket creation is disabled then log the bucket we skipped creating
    if config.skip_bucket_auto_create {
        // log the bucket we aren't creating
        println!("Skipping bucket creation: {bucket_name}");
        // just return instead of creating buckets
        return Ok(());
    }
    // build out the bucket creation config
    let mut bucket_config = CreateBucketConfiguration::builder();
    // if we have a region set then set the location con
    if let Some(region) = &config.region {
        // build our constraint
        let constraint = BucketLocationConstraint::from(region.as_str());
        // set our constraint
        bucket_config = bucket_config.location_constraint(constraint);
    }
    // build our bucket config
    let bucket_config = bucket_config.build();
    // attempt to create the bucket
    let response = client
        .create_bucket()
        .create_bucket_configuration(bucket_config)
        .bucket(bucket_name)
        .send()
        .await;
    match response {
        // bucket was created
        Ok(_) => {
            println!("Created S3 bucket {}", bucket_name);
            Ok(())
        }
        Err(error) => match error {
            SdkError::ServiceError(service_err) => match service_err.err() {
                // bucket already exists
                CreateBucketError::BucketAlreadyExists(msg) => Err(Error::new(format!(
                    "Failed to create bucket {}: {}",
                    bucket_name, msg
                ))),
                // Bucket already exists and we likely have permissions to write
                CreateBucketError::BucketAlreadyOwnedByYou(_msg) => {
                    println!("Bucket already exists: {}", bucket_name);
                    Ok(())
                }
                _ => Err(Error::new(format!(
                    "Failed to create bucket {}: {:?}",
                    bucket_name, service_err
                ))),
            },
            _ => Err(Error::new(format!(
                "Failed to create bucket {}: {}",
                bucket_name, error
            ))),
        },
    }
}

/// Create the S3 buckets required for a ThoriumCluster
///
/// # Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn create_all_buckets(meta: &ClusterMeta) -> Result<(), Error> {
    // get s3 portion of config
    let s3 = &meta.cluster.spec.config.thorium.s3;
    let config = &meta.cluster.spec.config.thorium;
    // get our s3 credentials
    let creds = Credentials::new(&s3.access_key, &s3.secret_token, None, None, "Thorium");
    // build our s3 config
    let mut s3_config_builder = aws_sdk_s3::config::Builder::new()
        .endpoint_url(&s3.endpoint)
        .credentials_provider(SharedCredentialsProvider::new(creds))
        .force_path_style(s3.use_path_style);
    // if we have a region set then add that to our config
    if let Some(region) = &s3.region {
        // set our region
        s3_config_builder =
            s3_config_builder.region(aws_types::region::Region::new(region.clone()));
    }
    // build our s3 config
    let s3_config = s3_config_builder.build();
    // build our s3 client from the s3 config
    let client = Client::from_conf(s3_config);
    // create all Thorium buckets
    create_bucket(s3, &client, &config.files.bucket).await?;
    create_bucket(s3, &client, &config.repos.bucket).await?;
    create_bucket(s3, &client, &config.attachments.bucket).await?;
    create_bucket(s3, &client, &config.results.bucket).await?;
    create_bucket(s3, &client, &config.ephemeral.bucket).await?;
    create_bucket(s3, &client, &config.graphics.bucket).await?;
    create_bucket(s3, &client, &config.reaction_cache.bucket).await?;
    Ok(())
}
