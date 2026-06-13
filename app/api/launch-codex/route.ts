import { exec } from 'child_process';

export async function POST() {
  try {
    // Trigger the launch-codex-tmux skill
    exec('claude launch-codex-tmux', (error, stdout, stderr) => {
      if (error) {
        console.error('Error launching Codex:', error);
      } else {
        console.log('Codex launched:', stdout);
      }
    });

    return Response.json(
      { success: true, message: 'Codex launcher initiated' },
      { status: 200 }
    );
  } catch (error) {
    console.error('Failed to launch Codex:', error);
    return Response.json(
      { success: false, error: 'Failed to launch Codex' },
      { status: 500 }
    );
  }
}
