import type { ReactNode } from "react";
import { CAT } from "./api";

export const Card = ({ title, children, wide }: {
  title: ReactNode; children: ReactNode; wide?: boolean;
}) => (
  <section className={`card${wide ? " wide" : ""}`}>
    <h2>{title}</h2>
    {children}
  </section>
);

export const Tile = ({ label, value, tone }: {
  label: string; value: ReactNode; tone?: "good" | "bad";
}) => (
  <div className="tile">
    <div className={`v ${tone ?? ""}`}>{value}</div>
    <div className="k">{label}</div>
  </div>
);

export const Table = ({ cols, rows, onRow }: {
  cols: string[]; rows: ReactNode[][]; onRow?: (i: number) => void;
}) => (
  <table>
    <thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
    <tbody>
      {rows.map((r, i) => (
        <tr key={i} className={onRow ? "click" : ""}
            onClick={() => onRow?.(i)}>
          {r.map((c, j) => <td key={j}>{c}</td>)}
        </tr>
      ))}
    </tbody>
  </table>
);

/** Horizontal bars with direct labels (identity/value never color-alone). */
export const HBars = ({ rows, color }: {
  rows: { k: string; v: number; txt?: string }[]; color?: string;
}) => {
  const max = Math.max(1, ...rows.map((r) => r.v));
  return (
    <div>
      {rows.map((r) => (
        <div className="hbar" key={r.k}>
          <div className="name" title={r.k}>{r.k}</div>
          <div className="track">
            <div className="fill"
                 style={{ width: `${(100 * r.v) / max}%`,
                          background: color ?? CAT[0] }} />
          </div>
          <div className="val">{r.txt ?? r.v}</div>
        </div>
      ))}
      {rows.length === 0 && <div className="note">no data</div>}
    </div>
  );
};

export const Note = ({ children }: { children: ReactNode }) => (
  <div className="note">{children}</div>
);
