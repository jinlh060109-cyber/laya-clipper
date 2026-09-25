// The one composition the clipper renders: a transparent layer of motion
// graphics, laid over a finished clip by ffmpeg. The AI director never writes
// code; it picks from the fixed components in graphics.tsx and fills their
// props (validated in clipper/director.py first).
import React from "react";
import {AbsoluteFill, Composition, Sequence, registerRoot} from "remotion";
import type {CalculateMetadataFunction} from "remotion";
import {GRAPHICS} from "./graphics";

export type Item = {component: string; from: number; to: number; props: Record<string, unknown>};
export type OverlayProps = {
  width: number; height: number; fps: number; duration: number;
  // Bands the captions and the hook title use, and where each `position`
  // puts a graphic (px from the top), all worked out in clipper/director.py.
  captionTop: number; captionBottom: number; titleBottom: number;
  slots: Record<string, number>; lowerThirdTop: number; accent: string; items: Item[];
};

const Overlay: React.FC<OverlayProps> = ({fps, duration, items, ...frame}) => {
  const total = Math.max(1, Math.round(duration * fps));
  return (
    <AbsoluteFill>
      {items.map((item, k) => {
        const Graphic = GRAPHICS[item.component];
        if (!Graphic) return null;
        const from = Math.max(0, Math.round(item.from * fps));
        const length = Math.max(1, Math.min(total, Math.round(item.to * fps)) - from);
        return (
          <Sequence key={k} from={from} durationInFrames={length} layout="none">
            <Graphic {...item.props} frame={{...frame, fps, duration}} />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};

const metadata: CalculateMetadataFunction<OverlayProps> = ({props}) => ({
  width: props.width, height: props.height, fps: props.fps,
  durationInFrames: Math.max(1, Math.round(props.duration * props.fps)),
});

const defaults: OverlayProps = {
  width: 1080, height: 1920, fps: 30, duration: 8, captionTop: 1235, captionBottom: 1402,
  titleBottom: 603, slots: {top: 328, upper: 808, center: 960, lower: 1681}, lowerThirdTop: 1621,
  accent: "#FFD400",
  items: [
    {component: "kinetic_text", from: 0.5, to: 3, props: {text: "1,162,500 XP", animation: "pop", position: "upper"}},
    {component: "counter", from: 3, to: 6, props: {value: 813, suffix: " games", label: "to finish"}},
    {component: "lower_third", from: 1, to: 5, props: {title: "Spike Rush", subtitle: "10-15 min a game"}},
    {component: "pointer", from: 5, to: 8, props: {x: 0.5, y: 0.45, label: "here"}},
    {component: "progress_bar", from: 0, to: 8, props: {}},
  ],
};

const Root: React.FC = () => (
  <Composition id="Overlay" component={Overlay} durationInFrames={240} fps={30}
               width={1080} height={1920} defaultProps={defaults} calculateMetadata={metadata} />
);

registerRoot(Root);
