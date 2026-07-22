//! Arguments for scoped token related Thorctl commands

use clap::Parser;

/// The commands to send to the tokens task handler
#[derive(Parser, Debug)]
pub enum Tokens {
    /// Create a new scoped token
    #[clap(version, author)]
    Create(CreateScopedToken),
    /// Get a table of all available scoped tokens
    #[clap(version, author)]
    Get(GetTokens),
    /// Update a scoped token
    #[clap(version, author)]
    Update(UpdateScopedToken),
    /// Delete scoped tokens
    #[clap(version, author)]
    Delete(DeleteScopedTokens),
    /// Describe specific scoped tokens, displaying details in JSON format
    #[clap(version, author)]
    Describe(DescribeTokens),
    /// Get info on the currently activated scoped token if one is active
    #[clap(version, author)]
    Current(CurrentScopedToken),
}

/// A command to create a new scoped token
#[derive(Parser, Debug)]
pub struct CreateScopedToken {
    /// The name of the scoped token to create
    pub name: String,
    /// The groups this scoped token is limited to
    #[clap(short, long, value_delimiter = ',', required = true)]
    pub groups: Vec<String>,
    /// The date this scoped token permanently expires making it ephemeral
    #[clap(long)]
    pub expires: Option<String>,
    /// The format expiration dates are in using chrono strftime specifiers
    ///     Example: The format of "2014-5-17T12:34:56" is "%Y-%m-%dT%H:%M:%S"
    ///     (see <https://docs.rs/chrono/latest/chrono/format/strftime>)
    #[clap(long, default_value = "%Y-%m-%dT%H:%M:%S", verbatim_doc_comment)]
    pub date_fmt: String,
}

/// A command to get a table of all available scoped tokens
#[derive(Parser, Debug)]
pub struct GetTokens {}

/// A command to update a scoped token
#[derive(Parser, Debug)]
pub struct UpdateScopedToken {
    /// The name of the scoped token to update
    pub name: String,
    #[clap(flatten)]
    pub opts: ScopedTokenUpdateOpts,
    /// The format expiration dates are in using chrono strftime specifiers
    ///     Example: The format of "2014-5-17T12:34:56" is "%Y-%m-%dT%H:%M:%S"
    ///     (see <https://docs.rs/chrono/latest/chrono/format/strftime>)
    #[clap(long, default_value = "%Y-%m-%dT%H:%M:%S", verbatim_doc_comment)]
    pub date_fmt: String,
}

/// The set of possible updates to a scoped token where at least one is set
#[derive(clap::Args, Debug, Clone)]
#[group(required = true, multiple = true)]
pub struct ScopedTokenUpdateOpts {
    /// A list of groups to add to this scoped tokens scope
    #[clap(long, value_delimiter = ',')]
    pub add_groups: Vec<String>,
    /// A list of groups to remove from this scoped tokens scope
    #[clap(long, value_delimiter = ',')]
    pub remove_groups: Vec<String>,
    /// The new date this scoped token permanently expires
    #[clap(long, conflicts_with = "clear_expires")]
    pub expires: Option<String>,
    /// Clear this scoped tokens expiration date making it no longer ephemeral
    #[clap(long, conflicts_with = "expires")]
    pub clear_expires: bool,
}

/// A command to delete scoped tokens
#[derive(Parser, Debug)]
pub struct DeleteScopedTokens {
    /// The names of the scoped tokens to delete
    #[clap(required = true)]
    pub names: Vec<String>,
}

/// A command to describe scoped tokens in full
#[derive(Parser, Debug)]
pub struct DescribeTokens {
    /// Any specific scoped tokens to describe (describes all when omitted)
    pub names: Vec<String>,
    /// Show each scoped tokens value
    #[clap(long)]
    pub show_token: bool,
    /// Output details in a condensed format (no formatting/whitespace)
    #[clap(long)]
    pub condensed: bool,
}

/// A command to activate a scoped token
#[derive(Parser, Debug)]
pub struct ActivateScopedToken {
    /// The name of the scoped token to activate
    pub name: String,
}

/// A command to get info on the currently activated scoped token
#[derive(Parser, Debug)]
pub struct CurrentScopedToken {
    /// Show this scoped tokens value
    #[clap(long)]
    pub show_token: bool,
}
