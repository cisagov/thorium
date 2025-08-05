//! A Request ID generator to allow us to track requests in logs

use http::Request;
use std::fmt;
use std::task::{Context, Poll};
use tower::layer::Layer;
use tower::Service;
use uuid::Uuid;

/// A uuidv4 to allow requests to be tracked in logs
#[derive(Serialize, Clone)]
pub struct ReqId(Uuid);

impl Default for ReqId {
    /// Creates a default [`ReqId`]
    fn default() -> Self {
        ReqId(Uuid::new_v4())
    }
}

impl fmt::Display for ReqId {
    /// Allow the request id to be displayed cleanly
    ///
    /// # Arguments
    ///
    /// * `fmt` - The formatter that is being used
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A request id middleware service
#[derive(Clone, Debug)]
pub struct ReqIdService<S> {
    inner: S,
}

impl<S> ReqIdService<S> {
    // Create a new Request ID middleware service
    pub fn new(inner: S) -> Self {
        Self { inner }
    }
}

impl<B, S: Service<Request<B>>> Service<Request<B>> for ReqIdService<S> {
    type Response = S::Response;
    type Error = S::Error;
    type Future = S::Future;

    /// check if our request service is done yet
    #[inline]
    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    /// Start handling a request
    fn call(&mut self, mut req: Request<B>) -> Self::Future {
        // generate a new request id and save it
        req.extensions_mut().insert(ReqId::default());
        // flag this middleware as finished
        self.inner.call(req)
    }
}

/// The layer to apply our request id middleware with
#[derive(Clone, Debug)]
pub struct ReqIdLayer;

impl<S> Layer<S> for ReqIdLayer {
    type Service = ReqIdService<S>;

    /// build this layer
    fn layer(&self, inner: S) -> Self::Service {
        ReqIdService { inner }
    }
}
