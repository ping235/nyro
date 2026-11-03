import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import clsx from "clsx";

type MarkdownViewProps = {
  text: string;
  /** "muted" renders the compact, low-contrast variant used by thinking blocks. */
  tone?: "default" | "muted";
  className?: string;
};

function flatten(children: ReactNode): string {
  if (children === null || children === undefined || children === false) return "";
  if (typeof children === "string" || typeof children === "number") return String(children);
  if (Array.isArray(children)) return children.map(flatten).join("");
  const node = children as { props?: { children?: ReactNode } };
  if (node.props?.children) return flatten(node.props.children);
  return "";
}

/** Markdown renderer shared by model answers and thinking blocks. */
export default function MarkdownView({ text, tone = "default", className }: MarkdownViewProps) {
  return (
    <div
      className={clsx(
        "model-test-markdown",
        tone === "muted" ? "model-test-markdown-muted text-xs" : "text-sm",
        className,
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ children, href }) => (
            <a href={href} target="_blank" rel="noreferrer noopener">
              {children}
            </a>
          ),
          pre: ({ children }) => <pre className="model-test-pre">{children}</pre>,
          code: ({ className: codeClassName, children }) => {
            const language = /language-([\w+-]+)/.exec(codeClassName ?? "")?.[1];
            const raw = flatten(children).replace(/\n$/, "");
            const isBlock = Boolean(language) || raw.includes("\n");
            if (!isBlock) {
              return <code className="model-test-code-inline">{children}</code>;
            }
            return (
              <code className="model-test-code-block" data-language={language ?? ""}>
                {raw}
              </code>
            );
          },
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
