import type { ReactNode } from "react";

type SectionShellProps = {
  title: string;
  caption: string;
  description: string;
  aside?: ReactNode;
  className?: string;
  children: ReactNode;
};

export function SectionShell({ title, caption, description, aside, className, children }: SectionShellProps) {
  const sectionClassName = ["studio-section", className].filter(Boolean).join(" ");

  return (
    <section className={sectionClassName}>
      <header className="studio-section__head">
        <div className="studio-section__title-block">
          <p className="studio-section__caption">{caption}</p>
          <h3>{title}</h3>
          <p className="studio-section__description">{description}</p>
        </div>
        {aside ? <div className="studio-section__aside">{aside}</div> : null}
      </header>

      <div>{children}</div>
    </section>
  );
}
