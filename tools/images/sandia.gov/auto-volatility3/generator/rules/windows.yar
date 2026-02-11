rule Windows_Memory_Image
{
    meta:
        description = "Detects common artifacts of a Windows memory image."
        author = "Google"
        version = "1.0"
        
    strings:
        // PDB paths for Windows kernel components
        // These will often appear in crash dumps and kernel memory.
        $pdb_ntos = "ntoskrnl.pdb" nocase
        $pdb_hal = "hal.pdb" nocase
        $pdb_win32k = "win32k.pdb" nocase
        
        // Windows Registry key strings
        $reg_key_hklm = "HKEY_LOCAL_MACHINE"
        $reg_key_nt = "Software\\Microsoft\\Windows NT\\CurrentVersion"
        
        // Key Windows process and system file names
        $process_explorer = "explorer.exe" nocase
        $process_lsass = "lsass.exe" nocase
        $process_csrss = "csrss.exe" nocase
        $system_kernel = "ntoskrnl.exe" nocase
        $system_dll = "kernel32.dll" nocase
        
        // Standard kernel driver path
        $driver_path = "\\System32\\drivers\\" nocase
        
    condition:
        // Match at least 4 of the specific Windows strings
        (4 of them) or
        // Alternative condition: a PDB string and a registry key
        ($pdb_ntos and $reg_key_hklm)
}

