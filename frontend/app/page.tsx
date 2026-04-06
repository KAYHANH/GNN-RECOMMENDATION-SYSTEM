"use client";

import { startTransition, useDeferredValue, useEffect, useState } from "react";

import {
  ProductCard,
  type RecommendationItem,
  getItemText,
  getProductTone
} from "../components/product-card";
import { SectionShell } from "../components/section-shell";

type RecommendationMode = "hybrid" | "gnn" | "semantic";

type RecommendationResponse = {
  customer_id: string;
  mode: RecommendationMode;
  recommendations: RecommendationItem[];
};

type SearchResponse = {
  query: string;
  results: RecommendationItem[];
};

type ExplainResponse = {
  customer_id: string;
  article_id: string;
  reasons: string[];
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

const QUICK_QUERIES = [
  "black summer dress",
  "wide leg denim",
  "minimal evening knit",
  "soft tailored layers"
] as const;

const MODE_DETAILS: Record<
  RecommendationMode,
  {
    label: string;
    copy: string;
  }
> = {
  hybrid: {
    label: "Hybrid",
    copy: "Fuse graph signals with semantic intent."
  },
  gnn: {
    label: "LightGCN",
    copy: "Let collaborative structure lead the ranking."
  },
  semantic: {
    label: "Semantic",
    copy: "Push query understanding to the front."
  }
};

const EXPERIENCE_PILLARS = [
  {
    index: "01",
    title: "Graph ranking with style memory",
    copy: "Use the interaction graph to surface pieces that feel aligned with what the customer already gravitates toward."
  },
  {
    index: "02",
    title: "Intent search without flattening taste",
    copy: "A live semantic lane turns natural language into product discovery without reducing the interface to a plain search box."
  },
  {
    index: "03",
    title: "Reasoning you can actually show",
    copy: "Every selected item opens a visible explanation trail, which makes the demo stronger for reviews, papers, and stakeholder walkthroughs."
  }
] as const;

const PROJECT_NOTES = [
  {
    title: "Demo-safe even when artifacts are missing",
    copy: "The UI stays alive with the repo's deterministic fallback behavior, so the project can still be shown while the full model stack is in progress."
  },
  {
    title: "Made for research storytelling",
    copy: "The layout gives you distinct surfaces for ranking, retrieval, and explanations, which makes screenshots and technical demos feel intentional."
  },
  {
    title: "Closer to a product than a notebook wrapper",
    copy: "Customer context, mode switching, query composition, and explanation tracing all live together in one cohesive studio instead of scattered controls."
  }
] as const;

const STACK_STEPS = [
  {
    index: "Layer 01",
    title: "Candidate graph",
    copy: "LightGCN with BPR handles collaborative recall and keeps the recommendation runway grounded in interaction structure.",
    footnote: "Ranking core"
  },
  {
    index: "Layer 02",
    title: "Semantic retrieval",
    copy: "Sentence-level understanding supports open-ended search prompts and rescues discovery when the graph is sparse.",
    footnote: "Search core"
  },
  {
    index: "Layer 03",
    title: "API and explainability",
    copy: "FastAPI exposes recommend, search, and explain endpoints so the frontend can feel live rather than mocked.",
    footnote: "Serving layer"
  },
  {
    index: "Layer 04",
    title: "Evaluation and MLOps",
    copy: "Temporal evaluation, MLflow, DVC, and CI move the project toward a research-ready workflow instead of a one-off experiment.",
    footnote: "Ops backbone"
  }
] as const;

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function getDominantValue(
  items: RecommendationItem[],
  pick: (item: RecommendationItem) => string,
  fallback: string
): string {
  const counts = new Map<string, number>();

  for (const item of items) {
    const value = pick(item).trim();
    if (!value) {
      continue;
    }

    counts.set(value, (counts.get(value) ?? 0) + 1);
  }

  let winner = fallback;
  let topCount = 0;

  for (const [value, count] of counts.entries()) {
    if (count > topCount) {
      winner = value;
      topCount = count;
    }
  }

  return winner;
}

function getSourceBlend(items: RecommendationItem[], fallback: string): string {
  const sources = Array.from(new Set(items.map((item) => item.source).filter(Boolean)));
  return sources.length > 0 ? sources.join(" / ") : fallback;
}

export default function Page() {
  const [customerId, setCustomerId] = useState("demo-customer");
  const [mode, setMode] = useState<RecommendationMode>("hybrid");
  const [query, setQuery] = useState("black summer dress");
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([]);
  const [searchResults, setSearchResults] = useState<RecommendationItem[]>([]);
  const [selectedItem, setSelectedItem] = useState<RecommendationItem | null>(null);
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);
  const [status, setStatus] = useState("Opening the atelier...");
  const [isFeedLoading, setIsFeedLoading] = useState(true);
  const [isSearchLoading, setIsSearchLoading] = useState(true);
  const [isExplanationLoading, setIsExplanationLoading] = useState(false);
  const deferredQuery = useDeferredValue(query);

  useEffect(() => {
    let cancelled = false;

    async function loadRecommendations() {
      try {
        setIsFeedLoading(true);
        setStatus("Refreshing the recommendation runway...");

        const payload = await getJson<RecommendationResponse>(
          `/recommend/${encodeURIComponent(customerId)}?k=6&mode=${encodeURIComponent(mode)}`
        );

        if (!cancelled) {
          startTransition(() => {
            setRecommendations(payload.recommendations);
            setStatus(`Loaded ${payload.recommendations.length} ranked looks for ${payload.customer_id}.`);
          });
        }
      } catch (error) {
        if (!cancelled) {
          setStatus(error instanceof Error ? error.message : "Failed to load recommendations.");
        }
      } finally {
        if (!cancelled) {
          setIsFeedLoading(false);
        }
      }
    }

    loadRecommendations();
    return () => {
      cancelled = true;
    };
  }, [customerId, mode]);

  useEffect(() => {
    if (recommendations.length === 0) {
      return;
    }

    setSelectedItem((current) => {
      if (current && recommendations.some((item) => item.article_id === current.article_id)) {
        return current;
      }

      return recommendations[0];
    });
  }, [recommendations]);

  useEffect(() => {
    if (!deferredQuery.trim()) {
      setSearchResults([]);
      setIsSearchLoading(false);
      return;
    }

    let cancelled = false;

    async function loadSearch() {
      try {
        setIsSearchLoading(true);
        const payload = await getJson<SearchResponse>(
          `/search?q=${encodeURIComponent(deferredQuery)}&k=6`
        );

        if (!cancelled) {
          startTransition(() => {
            setSearchResults(payload.results);
          });
        }
      } catch (error) {
        if (!cancelled) {
          setStatus(error instanceof Error ? error.message : "Failed to run semantic search.");
        }
      } finally {
        if (!cancelled) {
          setIsSearchLoading(false);
        }
      }
    }

    loadSearch();
    return () => {
      cancelled = true;
    };
  }, [deferredQuery]);

  useEffect(() => {
    if (searchResults.length === 0) {
      return;
    }

    setSelectedItem((current) => current ?? searchResults[0]);
  }, [searchResults]);

  useEffect(() => {
    if (!selectedItem) {
      setExplanation(null);
      return;
    }

    const activeItem = selectedItem;
    let cancelled = false;

    async function loadExplanation() {
      try {
        setIsExplanationLoading(true);
        setStatus(`Tracing the reasoning for ${getItemText(activeItem, "prod_name", activeItem.article_id)}...`);

        const payload = await getJson<ExplainResponse>(
          `/explain/${encodeURIComponent(customerId)}/${encodeURIComponent(activeItem.article_id)}`
        );

        if (!cancelled) {
          startTransition(() => {
            setExplanation(payload);
            setStatus(`Explanation ready for ${payload.article_id}.`);
          });
        }
      } catch (error) {
        if (!cancelled) {
          setStatus(error instanceof Error ? error.message : "Failed to load explanation.");
        }
      } finally {
        if (!cancelled) {
          setIsExplanationLoading(false);
        }
      }
    }

    loadExplanation();
    return () => {
      cancelled = true;
    };
  }, [customerId, selectedItem]);

  const liveItem = selectedItem ?? recommendations[0] ?? searchResults[0] ?? null;
  const liveItemName = liveItem ? getItemText(liveItem, "prod_name", liveItem.article_id) : "No article selected";
  const liveItemDescription = liveItem
    ? getItemText(liveItem, "detail_desc", "Choose a result to inspect why it belongs in the customer story.")
    : "Choose a result to inspect why it belongs in the customer story.";
  const liveItemColor = liveItem ? getItemText(liveItem, "colour_group_name", "Neutral tone") : "Neutral tone";
  const liveItemCategory = liveItem ? getItemText(liveItem, "product_group_name", "Fashion item") : "Fashion item";
  const recommendationTone = getDominantValue(
    recommendations,
    (item) => getItemText(item, "colour_group_name", ""),
    "Neutral edit"
  );
  const recommendationCategory = getDominantValue(
    recommendations,
    (item) => getItemText(item, "product_group_name", ""),
    "Mixed categories"
  );
  const sourceBlend = getSourceBlend(recommendations, MODE_DETAILS[mode].label);
  const leadTone = getProductTone(liveItemColor);
  const searchCountLabel = `${searchResults.length.toString().padStart(2, "0")} matches`;
  const recommendationCountLabel = `${recommendations.length.toString().padStart(2, "0")} ranked looks`;
  const searchEmptyMessage = deferredQuery.trim()
    ? "No semantic matches yet. Try a more specific silhouette, fabric, or occasion."
    : "Write a style prompt or tap one of the quick picks to start the semantic lane.";

  return (
    <main className="page-shell">
      <section className="hero">
        <div className="hero__media">
          <div className="hero__image" aria-hidden="true" />
          <div className="hero__wash" aria-hidden="true" />

          <div className="hero__nav">
            <p className="brand-mark">Fashion Graph Atelier</p>
            <div className="hero__live">
              <span className="hero__live-dot" />
              <span>{status}</span>
            </div>
          </div>

          <div className="hero__body">
            <div className="hero__content">
              <p className="eyebrow eyebrow--light">Advanced fashion recommender studio</p>
              <h1>Not a generic frontend. A showroom for search, ranking, and taste signals.</h1>
              <p className="hero__summary">
                This interface treats the project like a real fashion product: cinematic entry, live customer controls,
                separate discovery lanes, and a reasoning rail that keeps the model legible while you browse.
              </p>

              <div className="hero__actions">
                <a className="pill-button" href="#workspace">
                  Open the studio
                </a>
                <p className="hero__mode-note">{MODE_DETAILS[mode].copy}</p>
              </div>

              <div className="hero__meta-grid">
                <div className="hero__metric">
                  <span>Dominant tone</span>
                  <strong>{recommendationTone}</strong>
                </div>
                <div className="hero__metric">
                  <span>Runway mix</span>
                  <strong>{sourceBlend}</strong>
                </div>
                <div className="hero__metric">
                  <span>Lead category</span>
                  <strong>{recommendationCategory}</strong>
                </div>
              </div>
            </div>

            <aside className="hero__deck">
              <div className="deck__intro">
                <p className="eyebrow eyebrow--light">Live controls</p>
                <h2>Compose the customer story.</h2>
              </div>

              <label className="field">
                <span>Customer ID</span>
                <input value={customerId} onChange={(event) => setCustomerId(event.target.value)} />
              </label>

              <div className="field">
                <span>Ranking mode</span>
                <div className="mode-switch" role="group" aria-label="Recommendation mode">
                  {(
                    Object.entries(MODE_DETAILS) as Array<
                      [RecommendationMode, { label: string; copy: string }]
                    >
                  ).map(([modeKey, modeDetail]) => (
                    <button
                      key={modeKey}
                      type="button"
                      className={`mode-chip${mode === modeKey ? " is-active" : ""}`}
                      onClick={() => setMode(modeKey)}
                      aria-pressed={mode === modeKey}
                    >
                      <span className="mode-chip__label">{modeDetail.label}</span>
                      <span className="mode-chip__copy">{modeDetail.copy}</span>
                    </button>
                  ))}
                </div>
              </div>

              <label className="field">
                <span>Semantic prompt</span>
                <textarea
                  rows={4}
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Describe the look, mood, fabric, or occasion."
                />
              </label>

              <div className="field">
                <span>Quick picks</span>
                <div className="quick-picks">
                  {QUICK_QUERIES.map((quickQuery) => (
                    <button
                      key={quickQuery}
                      type="button"
                      className={`quick-pick${query === quickQuery ? " is-active" : ""}`}
                      onClick={() => setQuery(quickQuery)}
                    >
                      {quickQuery}
                    </button>
                  ))}
                </div>
              </div>
            </aside>
          </div>
        </div>
      </section>

      <section className="manifesto frame">
        {EXPERIENCE_PILLARS.map((pillar) => (
          <article key={pillar.index} className="manifesto__item">
            <p className="manifesto__index">{pillar.index}</p>
            <h2>{pillar.title}</h2>
            <p>{pillar.copy}</p>
          </article>
        ))}
      </section>

      <section id="workspace" className="workspace frame">
        <div className="workspace__intro">
          <div>
            <p className="eyebrow">Studio floor</p>
            <h2>Two discovery lanes and one reasoning rail.</h2>
          </div>
          <p>
            Search and recommendations stay separate on purpose, so you can see where direct user intent ends and
            personalized ranking begins. The selected product stays pinned on the right with explanation output and
            model-facing cues.
          </p>
        </div>

        <div className="workspace__grid">
          <SectionShell
            title="Semantic lane"
            caption="Intent retrieval"
            description="Natural-language discovery responding to the live prompt."
            aside={<span className="count-chip">{searchCountLabel}</span>}
          >
            <div className="product-stack">
              {searchResults.map((item) => (
                <ProductCard
                  key={`search-${item.article_id}`}
                  item={item}
                  eyebrow="Search match"
                  actionLabel="Inspect match"
                  selected={selectedItem?.article_id === item.article_id}
                  onAction={setSelectedItem}
                />
              ))}

              {searchResults.length === 0 ? (
                <p className="empty-state">
                  {isSearchLoading ? "Scanning the semantic catalog..." : searchEmptyMessage}
                </p>
              ) : null}
            </div>
          </SectionShell>

          <SectionShell
            title="Recommendation runway"
            caption="Personalized ranking"
            description="A tighter feed shaped by collaborative structure and mode selection."
            aside={<span className="count-chip">{recommendationCountLabel}</span>}
          >
            <div className="product-stack product-stack--offset">
              {recommendations.map((item) => (
                <ProductCard
                  key={`recommend-${item.article_id}`}
                  item={item}
                  eyebrow="Runway pick"
                  actionLabel="Open reasoning"
                  selected={selectedItem?.article_id === item.article_id}
                  onAction={setSelectedItem}
                />
              ))}

              {recommendations.length === 0 ? (
                <p className="empty-state">
                  {isFeedLoading ? "Refreshing the personalized runway..." : "Recommendations will appear here."}
                </p>
              ) : null}
            </div>
          </SectionShell>

          <aside className="insight-rail">
            <div className="insight-rail__sticky">
              <section
                className="insight-preview"
                style={{
                  backgroundImage: `linear-gradient(150deg, ${leadTone} 0%, rgba(21, 18, 16, 0.96) 72%)`
                }}
              >
                <p className="eyebrow eyebrow--light">Selected article</p>
                <h3>{liveItemName}</h3>
                <p>{liveItemDescription}</p>
                <div className="tag-row">
                  <span className="tag">{liveItemCategory}</span>
                  <span className="tag">{liveItemColor}</span>
                  <span className="tag">{liveItem ? liveItem.source : MODE_DETAILS[mode].label}</span>
                </div>
              </section>

              <section className="signal-panel">
                <div className="signal-row">
                  <span>Customer profile</span>
                  <strong>{customerId}</strong>
                </div>
                <div className="signal-row">
                  <span>Selected mode</span>
                  <strong>{MODE_DETAILS[mode].label}</strong>
                </div>
                <div className="signal-row">
                  <span>Recommendation mood</span>
                  <strong>{recommendationTone}</strong>
                </div>
                <div className="signal-row">
                  <span>Feed category</span>
                  <strong>{recommendationCategory}</strong>
                </div>
              </section>

              <section className="signal-panel">
                <p className="eyebrow">Explanation trace</p>
                {isExplanationLoading ? (
                  <p className="empty-state">Tracing hybrid reasoning for the selected look...</p>
                ) : explanation ? (
                  <ol className="reason-list">
                    {explanation.reasons.map((reason) => (
                      <li key={reason}>{reason}</li>
                    ))}
                  </ol>
                ) : (
                  <p className="empty-state">Choose a product from either lane to open the explanation trail.</p>
                )}
              </section>
            </div>
          </aside>
        </div>
      </section>

      <section className="lookbook frame">
        <div className="lookbook__image" aria-hidden="true" />

        <div className="lookbook__copy">
          <p className="eyebrow">Project lift</p>
          <h2>The project now reads like a product concept, not a template-wrapped API demo.</h2>
          <div className="note-list">
            {PROJECT_NOTES.map((note) => (
              <article key={note.title} className="note-item">
                <strong>{note.title}</strong>
                <p>{note.copy}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="systems frame">
        <div className="systems__header">
          <div>
            <p className="eyebrow">Advanced stack</p>
            <h2>Built to keep evolving beyond the frontend polish.</h2>
          </div>
          <p>
            The interface is only one layer. Under it, the project is already structured around retrieval, ranking,
            explainability, evaluation, and MLOps so you can keep pushing it toward a stronger final system.
          </p>
        </div>

        <div className="systems__grid">
          {STACK_STEPS.map((step) => (
            <article key={step.title} className="system-step">
              <span className="system-step__index">{step.index}</span>
              <h3>{step.title}</h3>
              <p>{step.copy}</p>
              <span className="system-step__footnote">{step.footnote}</span>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
