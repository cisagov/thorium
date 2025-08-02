//! Utility functions/macros shared across the Thorium client

/// Calculates the correct page size in case the limit in smaller
/// than the given page size
#[macro_export]
macro_rules! calculate_page_size {
    ($page_size:expr, $limit:expr) => {
        $limit.map_or_else(|| $page_size, |limit| std::cmp::min($page_size, limit))
    };
}
