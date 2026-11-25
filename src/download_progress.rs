use crate::GenerateProgress;
use huggingface_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use std::io::{self, Write};
use std::sync::{Arc, Mutex, OnceLock};

static IPC_OUTPUT_BUFFER: OnceLock<Arc<Mutex<Vec<u8>>>> = OnceLock::new();

#[derive(Clone)]
pub struct IpcStdout {
    buffer: Arc<Mutex<Vec<u8>>>,
}

pub fn ipc_stdout() -> IpcStdout {
    IpcStdout {
        buffer: Arc::clone(IPC_OUTPUT_BUFFER.get_or_init(|| Arc::new(Mutex::new(Vec::new())))),
    }
}

impl Write for IpcStdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut buffer = self
            .buffer
            .lock()
            .map_err(|_| io::Error::other("IPC stdout buffer lock is poisoned"))?;
        buffer.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut buffer = self
            .buffer
            .lock()
            .map_err(|_| io::Error::other("IPC stdout buffer lock is poisoned"))?;
        let mut stdout = io::stdout().lock();
        stdout.write_all(&buffer)?;
        stdout.flush()?;
        buffer.clear();
        Ok(())
    }
}

#[derive(Default)]
struct DisplayState {
    completed: u64,
    total: u64,
    last_line: String,
}

enum ProgressOutput {
    Terminal,
    Ipc(IpcStdout),
}

pub struct CliDownloadProgress {
    filename: String,
    output: ProgressOutput,
    state: Mutex<DisplayState>,
}

pub fn download_progress(filename: &str) -> CliDownloadProgress {
    let output = if crate::stderr_logging_enabled() {
        ProgressOutput::Terminal
    } else {
        ProgressOutput::Ipc(ipc_stdout())
    };
    CliDownloadProgress {
        filename: filename.to_string(),
        output,
        state: Mutex::new(DisplayState::default()),
    }
}

impl CliDownloadProgress {
    fn display(&self, completed: u64, total: u64, finished: bool) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        state.completed = completed.max(state.completed);
        state.total = total.max(state.total);
        let line = progress_line(&self.filename, state.completed, state.total, finished);
        if line == state.last_line {
            return;
        }
        state.last_line.clone_from(&line);
        match &self.output {
            ProgressOutput::Terminal => {
                let terminator = if finished { "\n" } else { "" };
                eprint!("\r\x1b[2K{line}{terminator}");
                let _ = io::stderr().flush();
            }
            ProgressOutput::Ipc(stdout) => {
                let progress = if state.total == 0 {
                    f32::from(finished)
                } else {
                    (state.completed as f64 / state.total as f64).clamp(0.0, 1.0) as f32
                };
                let message = GenerateProgress {
                    phase: "download".to_string(),
                    progress,
                    operation: format!("Downloading {}", self.filename),
                };
                let mut stdout = stdout.clone();
                let _ = crate::write_ipc_message(&mut stdout, &message);
                let _ = stdout.flush();
            }
        }
    }
}

impl ProgressHandler for CliDownloadProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        match event {
            DownloadEvent::Start { total_bytes, .. } => self.display(0, *total_bytes, false),
            DownloadEvent::Progress { files } => {
                if let Some(file) = files.iter().find(|file| file.filename == self.filename) {
                    self.display(file.bytes_completed, file.total_bytes, false);
                }
            }
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                ..
            } => self.display(*bytes_completed, *total_bytes, false),
            DownloadEvent::Complete => {
                let (completed, total) = self
                    .state
                    .lock()
                    .map(|state| (state.completed, state.total))
                    .unwrap_or((0, 0));
                self.display(completed.max(total), total, true);
            }
        }
    }
}

fn progress_line(filename: &str, completed: u64, total: u64, finished: bool) -> String {
    if total == 0 {
        let status = if finished { "ready" } else { "starting" };
        return format!("[download] {filename}: {status}");
    }
    let percent = (completed as f64 * 100.0 / total as f64).clamp(0.0, 100.0);
    format!(
        "[download] {filename}: {percent:5.1}% ({}/{})",
        format_bytes(completed),
        format_bytes(total)
    )
}

fn format_bytes(bytes: u64) -> String {
    const MEBIBYTE: f64 = 1024.0 * 1024.0;
    const GIBIBYTE: f64 = 1024.0 * MEBIBYTE;
    let bytes = bytes as f64;
    if bytes >= GIBIBYTE {
        format!("{:.2} GiB", bytes / GIBIBYTE)
    } else {
        format!("{:.1} MiB", bytes / MEBIBYTE)
    }
}

#[cfg(test)]
mod tests {
    use super::{format_bytes, ipc_stdout, progress_line};
    use std::io::Write;

    #[test]
    fn formats_download_progress() {
        assert_eq!(
            progress_line("model.bpk", 512 * 1024 * 1024, 1024 * 1024 * 1024, false),
            "[download] model.bpk:  50.0% (512.0 MiB/1.00 GiB)"
        );
        assert_eq!(
            progress_line("tokenizer.json", 0, 0, true),
            "[download] tokenizer.json: ready"
        );
    }

    #[test]
    fn formats_small_file_sizes() {
        assert_eq!(format_bytes(2 * 1024 * 1024), "2.0 MiB");
    }

    #[test]
    fn ipc_stdout_clones_share_message_buffering() {
        let mut first = ipc_stdout();
        let mut second = first.clone();
        first.write_all(&7_u64.to_le_bytes()).unwrap();
        second.write_all(b"message").unwrap();
        assert_eq!(
            first.buffer.lock().unwrap().as_slice(),
            [7_u64.to_le_bytes().as_slice(), b"message"].concat()
        );
        first.buffer.lock().unwrap().clear();
    }
}
