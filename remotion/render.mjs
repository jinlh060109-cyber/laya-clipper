// Renders the Overlay composition for the Python side (clipper/remotion.py).
//   node render.mjs video <props.json> <out.mov>        transparent ProRes 4444
//   node render.mjs stills <props.json> <outdir> t1,t2  transparent PNGs at those seconds
// The bundle is built once into build/ and reused while src/ is unchanged.
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";
import {bundle} from "@remotion/bundler";
import {ensureBrowser, openBrowser, renderMedia, renderStill, selectComposition} from "@remotion/renderer";

const here = path.dirname(fileURLToPath(import.meta.url));
const build = path.join(here, "build");

function newest(dir) {
  return Math.max(...fs.readdirSync(dir, {withFileTypes: true}).map((e) => {
    const p = path.join(dir, e.name);
    return e.isDirectory() ? newest(p) : fs.statSync(p).mtimeMs;
  }));
}

async function serveUrl() {
  const stamp = path.join(build, "index.html");
  if (fs.existsSync(stamp) && fs.statSync(stamp).mtimeMs > newest(path.join(here, "src"))) return build;
  return bundle({entryPoint: path.join(here, "src", "index.tsx"), outDir: build});
}

const [mode, propsFile, out, times] = process.argv.slice(2);
const inputProps = JSON.parse(fs.readFileSync(propsFile, "utf-8"));
await ensureBrowser();
const url = await serveUrl();
const browser = await openBrowser("chrome");
try {
  const composition = await selectComposition({serveUrl: url, id: "Overlay", inputProps, puppeteerInstance: browser});
  if (mode === "video") {
    let last = -1;
    await renderMedia({
      composition, serveUrl: url, inputProps, puppeteerInstance: browser, outputLocation: out,
      codec: "prores", proResProfile: "4444", imageFormat: "png", pixelFormat: "yuva444p10le",
      concurrency: "50%", muted: true,
      onProgress: ({progress}) => {
        const pct = Math.floor(progress * 100);
        if (pct >= last + 10) { last = pct; console.log(`progress ${pct}`); }
      },
    });
  } else if (mode === "stills") {
    fs.mkdirSync(out, {recursive: true});
    for (const t of times.split(",").map(Number)) {
      const frame = Math.min(composition.durationInFrames - 1, Math.max(0, Math.round(t * composition.fps)));
      await renderStill({composition, serveUrl: url, inputProps, puppeteerInstance: browser, frame,
                         imageFormat: "png", output: path.join(out, `${t.toFixed(2)}.png`)});
    }
  } else {
    throw new Error(`Unknown mode ${mode}`);
  }
  console.log("done");
} finally {
  await browser.close({silent: true});
}
