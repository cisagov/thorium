//! Parse av3 results into processes

use thorium::models::{WindowsProcessEntity, WindowsProcessTreeEntity};
use thorium::{CtlConf, Error, Thorium};

use chrono::{DateTime, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::fs;

//#[derive(Debug, Clone, Default, Serialize, Deserialize)]
//pub struct WindowsProcessEntity {
//    /// This processes id
//    pub pid: u64,
//    /// This processes parent PID
//    pub parent_pid: Option<u64>,
//    /// The name of the executable for this processes
//    pub name: Option<String>,
//    /// The path to this executable
//    pub image_path: Option<String>,
//    /// The full cmd for this process
//    pub command: Option<String>,
//    /// The offset for this process
//    pub offset: Option<u64>,
//    /// the number of threads this process spawned
//    pub threads: Option<u32>,
//    /// The number of handles this process had open
//    pub handles: Option<u32>,
//    /// Whether this process is using the wow64 emulator or not
//    pub is_wow64: Option<bool>,
//    /// The session id for this process
//    pub session_id: Option<u32>,
//    /// When this process was spawned (not created in Thorium)
//    pub create_time: Option<DateTime<Utc>>,
//    /// When this process exited
//    pub exit_time: Option<DateTime<Utc>>,
//}
//
//#[derive(Debug)]
//pub enum Error {
//    Io(std::io::Error),
//}

//impl From<std::io::Error> for Error {
//    fn from(value: std::io::Error) -> Self {
//        Self::Io(value)
//    }
//}

pub async fn parse_windows_pstree_file(
    path: impl AsRef<std::path::Path>,
) -> Result<Vec<WindowsProcessEntity>, Error> {
    let content = fs::read_to_string(path).await?;
    Ok(parse_windows_pstree_str(&content))
}

pub fn parse_windows_pstree_str(input: &str) -> Vec<WindowsProcessEntity> {
    input.lines().filter_map(parse_process_line).collect()
}

fn parse_process_line(line: &str) -> Option<WindowsProcessEntity> {
    let line = line.trim();

    if line.is_empty() {
        return None;
    }

    // Skip file wrappers / markdown / headers
    if line.starts_with("<<<")
        || line.starts_with("```")
        || line.starts_with("Volatility 3 Framework")
        || line.starts_with("PID\t")
        || line.starts_with("PID ")
    {
        return None;
    }

    // Split on tabs first; fallback to whitespace if needed.
    let tab_parts: Vec<&str> = line.split('\t').collect();

    if tab_parts.len() >= 12 {
        return parse_tabbed_process_line(&tab_parts);
    }

    // Fallback for non-tabbed lines: not ideal for Cmd/Path fields with spaces,
    // but included for resilience.
    parse_whitespace_process_line(line)
}

fn parse_tabbed_process_line(parts: &[&str]) -> Option<WindowsProcessEntity> {
    if parts.len() < 12 {
        return None;
    }

    let pid = parse_tree_pid(parts[0])?;
    let parent_pid = parse_optional_u64(parts[1]);
    let name = parse_optional_string(parts[2]);
    let offset = parse_optional_hex_u64(parts[3]);
    let threads = parse_optional_u32(parts[4]);
    let handles = parse_optional_u32(parts[5]);
    let session_id = parse_optional_u32(parts[6]);
    let is_wow64 = parse_optional_bool(parts[7]);
    let create_time = parse_optional_datetime(parts[8]);
    let exit_time = parse_optional_datetime(parts[9]);

    // The actual layout from the sample is:
    // [0] PID
    // [1] PPID
    // [2] ImageFileName
    // [3] Offset(V)
    // [4] Threads
    // [5] Handles
    // [6] SessionId
    // [7] Wow64
    // [8] CreateTime
    // [9] ExitTime
    // [10] Audit
    // [11] Cmd
    // [12] Path
    //
    // Sometimes extra tabs may appear, so be defensive.
    let image_path = parts.get(12).and_then(|s| parse_optional_string(s));
    let command = parts.get(11).and_then(|s| parse_optional_string(s));
    let path_col = parts.get(12).and_then(|s| parse_optional_string(s));

    Some(WindowsProcessEntity {
        pid,
        parent_pid,
        name,
        image_path: image_path.or(path_col.clone()),
        command,
        offset,
        threads,
        handles,
        is_wow64,
        session_id,
        create_time,
        exit_time,
    })
}

fn parse_whitespace_process_line(line: &str) -> Option<WindowsProcessEntity> {
    let mut iter = line.split_whitespace();

    let pid = parse_tree_pid(iter.next()?)?;
    let parent_pid = parse_optional_u64(iter.next()?);
    let name = parse_optional_string(iter.next()?);
    let offset = parse_optional_hex_u64(iter.next()?);
    let threads = parse_optional_u32(iter.next()?);
    let handles = parse_optional_u32(iter.next()?);
    let session_id = parse_optional_u32(iter.next()?);
    let is_wow64 = parse_optional_bool(iter.next()?);

    // CreateTime = 3 tokens: YYYY-MM-DD HH:MM:SS.ffffff UTC
    let create_date = iter.next()?;
    let create_time_part = iter.next()?;
    let create_tz = iter.next()?;
    let create_time = parse_optional_datetime(&format!(
        "{} {} {}",
        create_date, create_time_part, create_tz
    ));

    // ExitTime = either N/A or 3 tokens
    let next = iter.next()?;
    let exit_time = if next == "N/A" {
        None
    } else {
        let t2 = iter.next()?;
        let t3 = iter.next()?;
        parse_optional_datetime(&format!("{} {} {}", next, t2, t3))
    };

    Some(WindowsProcessEntity {
        pid,
        parent_pid,
        name,
        image_path: None,
        command: None,
        offset,
        threads,
        handles,
        is_wow64,
        session_id,
        create_time,
        exit_time,
    })
}

fn parse_tree_pid(raw: &str) -> Option<u64> {
    let cleaned = raw.trim_start_matches('*').trim();
    cleaned.parse::<u64>().ok()
}

fn parse_optional_u64(raw: &str) -> Option<u64> {
    match raw.trim() {
        "-" | "N/A" => None,
        s => s.parse::<u64>().ok(),
    }
}

fn parse_optional_u32(raw: &str) -> Option<u32> {
    match raw.trim() {
        "-" | "N/A" => None,
        s => s.parse::<u32>().ok(),
    }
}

fn parse_optional_hex_u64(raw: &str) -> Option<u64> {
    let s = raw.trim();
    if s == "-" || s == "N/A" {
        return None;
    }

    let s = s.strip_prefix("0x").unwrap_or(s);
    u64::from_str_radix(s, 16).ok()
}

fn parse_optional_bool(raw: &str) -> Option<bool> {
    match raw.trim() {
        "True" | "true" => Some(true),
        "False" | "false" => Some(false),
        "-" | "N/A" => None,
        _ => None,
    }
}

fn parse_optional_string(raw: &str) -> Option<String> {
    let s = raw.trim();
    match s {
        "" | "-" | "N/A" => None,
        _ => Some(s.to_string()),
    }
}

fn parse_optional_datetime(raw: &str) -> Option<DateTime<Utc>> {
    let s = raw.trim();
    if s == "-" || s == "N/A" {
        return None;
    }

    // Example: 2024-02-01 19:48:22.000000 UTC
    let s = s.strip_suffix(" UTC").unwrap_or(s);

    let naive = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f").ok()?;
    Some(DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc))
}

pub async fn run() -> Result<(), Error> {
    // load ctl conf
    let config = CtlConf::from_path("/Users/mcarson/.thorium/config.yml")?;
    // build our Thorium client based on our config
    let thorium = Thorium::from_ctl_conf(config.clone()).await?;
    // read in our processes
    let processes = parse_windows_pstree_file("windows_pstree.txt")
        .await
        .unwrap();

    println!("parsed {} processes", processes.len());

    // build a toy sigma rule
    let rule_yaml = r#"
    title: A test rule
    logsource:
        category: test
    detection:
        selection_1:
            threads|contains: '11'
            command|endswith:
                - 'MaxRequestThreads=16'
        selection_2:
            command|contains: 'alex'
        condition: 1 of selection_*
    "#;

    let rule = sigma_rust::rule_from_yaml(rule_yaml).unwrap();
    //let event = sigma_rust::event_from_json(
    //    serialize!(), //r#"{"TargetFilename": "C:\\temp\\file.au3", "Image": "C:\\temp\\autoit4.exe", "Event": {"ID": 42}}"#,
    //)
    //.unwrap();
    //println!("RULE HIT!: {}", rule.is_match(&event));
    let mut builder =
        WindowsProcessTreeEntity::builder("Toy Process Tree").tool("auto-volatility3");
    for proc in processes.into_iter() {
        let event = sigma_rust::event_from_json(&serde_json::to_string(&proc).unwrap()).unwrap();
        if rule.is_match(&event) {
            println!("RULE HIT -> {:?} / {:?}", proc.name, proc.command);
        }
        // add this process
        builder.add_mut(proc);
        //println!("{:#?}", proc);
        //println!(">>>>> RULE HIT!: {}\n\n", rule.is_match(&event));
    }
    // add this process tree in Thorium
    builder.create(&["corn".to_owned()], &thorium).await?;
    Ok(())
}
