//! An agent to write a summary for data in Thorium

use thorium::Error;

use super::{AiSupport, ThorChat};
use crate::args::ai::Summary;

pub async fn handle<A: AiSupport>(thorchat: &mut ThorChat<A>, cmd: &Summary) -> Result<(), Error> {
    // build the question to ask
    let question = format!(
        "Summarize the following sample in Thorium: {}.

         Do not include the file hashes we are asking about. In the \
         main summary do not mention the results that are available. Make sure \
         to take into account tags, comments, and results into the global summary \
         and try to explain what they mean in your summary. The top level main \
         summary can be longer, up to multiple paragraphs, if needed to ensure \
         an accurate and descriptive summary.

         In the tags summary try to format the tags in a list format and cover \
         the only the most important tags, what they mean, and why you think \
         they are important. You do not need to include all tags only the ones \
         you think are important.

         In the results summary organize your result summaries by tool name. \
         For each result only cover the important results and make sure to explain \
         what is important and why. Use lists when possible for readability.

         Under no circumstances should your final summary be in json. \
         Your summary should be in nicely formatted markdown that \
         emphasizes readability in the following format:

         ### Summary

         ### Tags Summary

         ### Results Summary",
        cmd.target,
    );
    // enable debug mode if needed
    thorchat.ai.debug_mode(cmd.debug);
    // only enable the relevant tools
    thorchat.enable_tool("get_sample");
    thorchat.enable_tool("get_sample_results");
    // ask our ai to summarize this hash and print its response
    match thorchat.ask(question).await? {
        Some(response) => Ok(println!("\n\n{response}")),
        None => Err(Error::new("AI returned no response")),
    }
}
