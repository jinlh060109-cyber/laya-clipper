// The graphics the AI director can place. Every one animates from
// useCurrentFrame() only (CSS animations do not render in Remotion), draws
// nothing behind itself, and stays out of the caption band.
import React from "react";
import {Easing, interpolate, spring, useCurrentFrame, useVideoConfig} from "remotion";

type Frame = {width: number; height: number; fps: number; duration: number;
              captionTop: number; captionBottom: number; titleBottom: number;
              slots: Record<string, number>; lowerThirdTop: number; accent: string};
type P<T> = T & {frame: Frame};

const FONT = "'Arial Black', 'Segoe UI', Arial, sans-serif";
const clamp = {extrapolateLeft: "clamp", extrapolateRight: "clamp"} as const;

// Where a text graphic sits: the slot's centre (worked out in Python for the
// layout), moved off the captions if it would touch them.
function topFor(position: string | undefined, f: Frame, boxHeight: number): number {
  const centre = f.slots[position ?? "upper"] ?? f.slots.upper ?? f.height * 0.35;
  let top = centre - boxHeight / 2;
  if (top < f.captionBottom + 10 && top + boxHeight > f.captionTop - 10) {
    top = centre < (f.captionTop + f.captionBottom) / 2 ? f.captionTop - boxHeight - 20 : f.captionBottom + 20;
  }
  return Math.min(Math.max(top, 30), f.height - boxHeight - 30);
}

// In and out: every graphic enters and leaves instead of popping on and off.
function useInOut(outFrames = 8) {
  const frame = useCurrentFrame();
  const {fps, durationInFrames} = useVideoConfig();
  const enter = spring({frame, fps, config: {damping: 14, stiffness: 180}});
  const exit = interpolate(frame, [durationInFrames - outFrames, durationInFrames], [1, 0], clamp);
  return {frame, fps, enter, exit, length: durationInFrames};
}

const stroke = (px: number) => ({
  WebkitTextStroke: `${px}px black`, paintOrder: "stroke fill" as const,
  textShadow: "0 6px 18px rgba(0,0,0,0.55)",
});

export const KineticText: React.FC<P<{text: string; animation?: string; position?: string}>> = (
  {text, animation = "pop", position, frame: f}) => {
  const {frame, fps, enter, exit} = useInOut();
  const size = Math.round(f.width * (text.length > 14 ? 0.072 : 0.095));
  const top = topFor(position, f, size * 2.4);
  let transform = `scale(${interpolate(enter, [0, 1], [0.4, 1])})`;
  let shown = text;
  if (animation === "slide") transform = `translateX(${interpolate(enter, [0, 1], [-f.width, 0])}px)`;
  if (animation === "typewriter") {
    transform = "none";
    shown = text.slice(0, Math.ceil(interpolate(frame, [0, Math.max(6, text.length * 1.6)], [0, text.length], clamp)));
  }
  if (animation === "punch") {
    const beat = spring({frame, fps, config: {damping: 6, stiffness: 260}});
    transform = `scale(${interpolate(beat, [0, 1], [1.8, 1])}) rotate(${interpolate(beat, [0, 1], [-8, -2])}deg)`;
  }
  return (
    <div style={{position: "absolute", left: 0, right: 0, top, display: "flex", justifyContent: "center",
                 opacity: exit, transform, padding: "0 60px"}}>
      <div style={{fontFamily: FONT, fontWeight: 900, fontSize: size, lineHeight: 1.1, textAlign: "center",
                   color: f.accent, ...stroke(Math.max(6, size / 9))}}>{shown}</div>
    </div>
  );
};

export const Counter: React.FC<P<{value: number; prefix?: string; suffix?: string; label?: string;
                                  decimals?: number; position?: string}>> = ({value, prefix = "", suffix = "", label, decimals = 0, position, frame: f}) => {
  const {frame, fps, enter, exit} = useInOut();
  const t = interpolate(frame, [0, Math.round(fps * 1.2)], [0, 1], {...clamp, easing: Easing.out(Easing.cubic)});
  const n = (value * t).toLocaleString("en-US", {minimumFractionDigits: decimals, maximumFractionDigits: decimals});
  const size = Math.round(f.width * 0.12);
  const top = topFor(position ?? "center", f, size * 2.2);
  return (
    <div style={{position: "absolute", left: 0, right: 0, top, textAlign: "center", opacity: exit,
                 transform: `translateY(${interpolate(enter, [0, 1], [60, 0])}px)`}}>
      <div style={{fontFamily: FONT, fontWeight: 900, fontSize: size, color: "white", ...stroke(size / 9)}}>
        {prefix}<span style={{color: f.accent}}>{n}</span>{suffix}
      </div>
      {label ? <div style={{fontFamily: FONT, fontSize: size * 0.34, color: "white", marginTop: 6,
                            ...stroke(5), opacity: enter}}>{label}</div> : null}
    </div>
  );
};

export const LowerThird: React.FC<P<{title: string; subtitle?: string}>> = ({title, subtitle, frame: f}) => {
  const {enter, exit} = useInOut(10);
  const size = Math.round(f.width * 0.05);
  const top = f.lowerThirdTop;
  const x = interpolate(enter, [0, 1], [-f.width * 0.9, 0]);
  return (
    <div style={{position: "absolute", left: 50, top, transform: `translateX(${x}px)`, opacity: exit,
                 display: "flex", alignItems: "stretch"}}>
      <div style={{width: 14, background: f.accent, borderRadius: 4}} />
      <div style={{background: "rgba(10,10,14,0.82)", padding: `${size * 0.35}px ${size * 0.6}px`,
                   borderRadius: "0 12px 12px 0"}}>
        <div style={{fontFamily: FONT, fontWeight: 900, fontSize: size, color: "white"}}>{title}</div>
        {subtitle ? <div style={{fontFamily: "Arial, sans-serif", fontSize: size * 0.62, color: "#d8d8d8",
                                 marginTop: 4}}>{subtitle}</div> : null}
      </div>
    </div>
  );
};

// A ring and an arrow on something in the picture; the director finds (x, y)
// by looking at the frames.
export const Pointer: React.FC<P<{x: number; y: number; label?: string}>> = ({x, y, label, frame: f}) => {
  const {frame, fps, enter, exit} = useInOut();
  const cx = x * f.width, cy = y * f.height;
  const r = f.width * 0.08;
  const pulse = 1 + 0.06 * Math.sin((frame / fps) * Math.PI * 2);
  const circumference = 2 * Math.PI * r;
  const labelAbove = cy - r - 150 > 0 && !(cy + r + 130 < f.captionTop);
  return (
    <div style={{position: "absolute", inset: 0, opacity: exit}}>
      <svg width={f.width} height={f.height} style={{position: "absolute", inset: 0}}>
        <circle cx={cx} cy={cy} r={r * pulse} fill="none" stroke={f.accent} strokeWidth={12}
                strokeDasharray={circumference} strokeDashoffset={circumference * (1 - enter)}
                style={{filter: "drop-shadow(0 0 8px rgba(0,0,0,0.7))"}} />
      </svg>
      {label ? (
        <div style={{position: "absolute", left: 0, right: 0, top: labelAbove ? cy - r - 130 : cy + r + 30,
                     textAlign: "center", fontFamily: FONT, fontWeight: 900, fontSize: f.width * 0.06,
                     color: "white", opacity: enter, ...stroke(7)}}>
          {labelAbove ? `${label} ↓` : `↑ ${label}`}
        </div>
      ) : null}
    </div>
  );
};

// Fills over the whole time it is shown (place it from 0 to the clip's end).
export const ProgressBar: React.FC<P<{position?: string}>> = ({position = "top", frame: f}) => {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  const done = interpolate(frame, [0, durationInFrames - 1], [0, 1], clamp);
  const top = position === "bottom" ? f.height - 18 : 0;
  return (
    <div style={{position: "absolute", left: 0, top, width: f.width, height: 18, background: "rgba(0,0,0,0.35)"}}>
      <div style={{width: `${done * 100}%`, height: "100%", background: f.accent}} />
    </div>
  );
};

// A sticker: a short word in a tilted badge, for a reaction beat.
export const Sticker: React.FC<P<{text: string; x?: number; y?: number}>> = ({text, x = 0.72, y = 0.3, frame: f}) => {
  const {frame, fps, exit} = useInOut();
  const pop = spring({frame, fps, config: {damping: 7, stiffness: 220}});
  const size = f.width * 0.06;
  const top = Math.min(Math.max(y * f.height, 30), f.height - size * 2.5);
  return (
    <div style={{position: "absolute", left: x * f.width, top, opacity: exit,
                 transform: `translate(-50%, 0) scale(${pop}) rotate(-7deg)`}}>
      <div style={{background: f.accent, color: "black", fontFamily: FONT, fontWeight: 900, fontSize: size,
                   padding: `${size * 0.25}px ${size * 0.5}px`, borderRadius: 16, border: "6px solid black",
                   boxShadow: "0 10px 24px rgba(0,0,0,0.5)", whiteSpace: "nowrap"}}>{text}</div>
    </div>
  );
};

export const GRAPHICS: Record<string, React.FC<any>> = {
  kinetic_text: KineticText, counter: Counter, lower_third: LowerThird,
  pointer: Pointer, progress_bar: ProgressBar, sticker: Sticker,
};
