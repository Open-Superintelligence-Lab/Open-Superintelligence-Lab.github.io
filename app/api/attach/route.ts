import { execFile } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);

// AppleScript: focus an existing Terminal window already attached to this tmux
// session (identified by a custom window title we set), or open a new Terminal
// window and attach. `sess` arrives as the first run-argument so the session
// name can't be interpolated into the script.
const SCRIPT = `on run argv
  set sess to item 1 of argv
  tell application "Terminal"
    activate
    set foundWin to missing value
    repeat with w in windows
      try
        if (custom title of w) is sess then
          set foundWin to w
          exit repeat
        end if
      end try
    end repeat
    if foundWin is not missing value then
      set index of foundWin to 1
    else
      do script "tmux attach -t " & quoted form of sess
      set custom title of front window to sess
    end if
  end tell
end run`;

export async function POST(req: Request) {
  let name = '';
  try {
    ({ name } = await req.json());
  } catch {
    name = '';
  }

  // tmux session names: letters, digits, dash, underscore, dot.
  if (!name || !/^[A-Za-z0-9._-]+$/.test(name)) {
    return Response.json({ success: false, error: 'invalid session name' }, { status: 400 });
  }

  try {
    await execFileAsync('/usr/bin/osascript', ['-e', SCRIPT, name], { timeout: 15_000 });
    return Response.json({ success: true, name }, { status: 200 });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error('Failed to attach to tmux session:', message);
    return Response.json({ success: false, name, error: message }, { status: 500 });
  }
}
