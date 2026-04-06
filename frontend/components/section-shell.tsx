import type { ReactNode } from "react";

type SectionShellProps = {
  title: string;
  subtitle: string;
  meta?: string;
  children: ReactNode;
};

export function SectionShell({ title, subtitle, meta, children }: SectionShellProps) {
  return (
    <section className="panel-section">
      <header className="panel-section__header">
        <div>
          <p className="panel-section__subtitle">{subtitle}</p>
          <h2>{title}</h2>
        </div>
        {meta ? <span className="panel-section__meta">{meta}</span> : null}
      </header>
      {children}
    </section>
  );
}
