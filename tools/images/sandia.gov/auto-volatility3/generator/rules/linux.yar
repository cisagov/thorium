rule Linux_Memory_Image
{
    meta:
        description = "Detects common artifacts of a Linux memory image."
        author = "Google"
        version = "1.0"

    strings:
        // Kernel version string patterns
        // E.g., Linux version 5.15.0-84-generic
        $kernel_version_major = "Linux version"
        $kernel_version_generic = "generic"
        
        // Standard Linux library and utility strings
        $lib_c = "libc.so"
        $lib_ld = "ld-linux.so"
        $lib_pthread = "libpthread.so"
        
        // Kernel-related structures and symbols
        $kernel_symbol = "__this_module"
        $module_init = "module_init"
        
        // Standard filesystem paths and process names
        $proc_path = "/proc/self/cmdline"
        $bash_history = ".bash_history"
        $cron_config = "/etc/crontab"
        
    condition:
        // Match at least 4 of the specific Linux strings
        (4 of them) or
        // Alternative condition: a kernel string and a library name
        ($kernel_version_major and $lib_c)
}
