const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');

async function capture() {
    const chromePath = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
    const chrome = spawn(chromePath, [
        '--headless=new',
        '--remote-debugging-port=9222',
        '--use-gl=angle',
        '--enable-webgl',
        '--window-size=1280,720',
        'http://127.0.0.1:8080/index.html?hide_overlay=1'
    ]);

    // Wait 2s for chrome to start
    await new Promise(r => setTimeout(r, 2000));

    // Get websocket debugger URL
    const targets = await new Promise((resolve, reject) => {
        http.get('http://127.0.0.1:9222/json', (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => resolve(JSON.parse(data)));
        }).on('error', reject);
    });

    const pageTarget = targets.find(t => t.type === 'page');
    if (!pageTarget) {
        console.error('No page target found');
        chrome.kill();
        return;
    }

    const ws = new WebSocket(pageTarget.webSocketDebuggerUrl);

    ws.onopen = async () => {
        // Wait 3 seconds for Three.js GLB load and first frames to render
        await new Promise(r => setTimeout(r, 3500));

        // Request screenshot
        ws.send(JSON.stringify({
            id: 1,
            method: 'Page.captureScreenshot',
            params: { format: 'png' }
        }));
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.id === 1 && msg.result && msg.result.data) {
            fs.writeFileSync('test/actual_viewer_render.png', Buffer.from(msg.result.data, 'base64'));
            console.log('Successfully captured test/actual_viewer_render.png!');
            ws.close();
            chrome.kill();
            process.exit(0);
        }
    };
}

capture().catch(err => {
    console.error('Capture error:', err);
    process.exit(1);
});
