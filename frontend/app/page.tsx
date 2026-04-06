"use client";

import { startTransition, useDeferredValue, useEffect, useState } from "react";

import {
  ProductCard,
  ProductMedia,
  type RecommendationItem,
  getItemText,
  getProductTone
} from "../components/product-card";
import { SectionShell } from "../components/section-shell";

type RecommendationMode = "hybrid" | "gnn" | "semantic";

type ServiceSnapshot = {
  artifacts: {
    articles_ready: boolean;
    transactions_ready: boolean;
    images_ready: boolean;
    semantic_index_ready: boolean;
    semantic_ids_ready: boolean;
    user_embeddings_ready: boolean;
    item_embeddings_ready: boolean;
    user_mapping_ready: boolean;
    item_mapping_ready: boolean;
    reranker_ready: boolean;
  };
  engines: {
    graph_ready: boolean;
    semantic_ready: boolean;
    reranker_ready: boolean;
    fallback_active: boolean;
  };
  catalog: {
    article_count: number;
    interaction_count: number;
    customer_count: number;
    image_count: number;
    sample_data_active: boolean;
  };
};

type ResponseMeta = {
  generated_at: string;
  environment: string;
  snapshot: ServiceSnapshot;
};

type RecommendationResponse = {
  customer_id: string;
  mode: RecommendationMode;
  recommendations: RecommendationItem[];
  meta: ResponseMeta;
};

type SearchResponse = {
  query: string;
  results: RecommendationItem[];
  meta: ResponseMeta;
};

type ExplainResponse = {
  customer_id: string;
  article_id: string;
  reasons: string[];
  meta: ResponseMeta;
};

type HealthResponse = {
  status: string;
  snapshot?: ServiceSnapshot | null;
  api: {
    name: string;
    version: string;
    environment: string;
    docs_url?: string | null;
    redoc_url?: string | null;
  };
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

const QUICK_QUERIES = [
  "black summer dress",
  "wide leg denim",
  "linen shirt for summer",
  "soft knit neutral layers"
] as const;

const MODE_DETAILS: Record<RecommendationMode, { label: string; copy: string }> = {
  hybrid: {
    label: "Hybrid",
    copy: "LightGCN candidates blended with semantic relevance."
  },
  gnn: {
    label: "LightGCN",
    copy: "Collaborative ranking only."
  },
  semantic: {
    label: "Semantic",
    copy: "Intent-driven retrieval emphasis."
  }
};

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function formatNumber(value: number | undefined): string {
  if (value === undefined) {
    return "-";
  }
  return new Intl.NumberFormat("en-US").format(value);
}

function getFeedLabel(snapshot: ServiceSnapshot | null): string {
  if (!snapshot) {
    return "Connecting";
  }

  if (snapshot.catalog.sample_data_active) {
    return "Fallback catalog";
  }

  return "H&M Kaggle catalog";
}

function getImageStatus(snapshot: ServiceSnapshot | null): string {
  if (!snapshot) {
    return "Checking images";
  }

  if (snapshot.artifacts.images_ready) {
    return `${formatNumber(snapshot.catalog.image_count)} local images`;
  }

  return "Images folder missing";
}

export default function Page() {
  const [customerId, setCustomerId] = useState("demo-customer");
  const [mode, setMode] = useState<RecommendationMode>("hybrid");
  const [query, setQuery] = useState("black summer dress");
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([]);
  const [searchResults, setSearchResults] = useState<RecommendationItem[]>([]);
  const [selectedItem, setSelectedItem] = useState<RecommendationItem | null>(null);
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);
  const [snapshot, setSnapshot] = useState<ServiceSnapshot | null>(null);
  const [status, setStatus] = useState("Loading workspace...");
  const [readiness, setReadiness] = useState("loading");
  const [isFeedLoading, setIsFeedLoading] = useState(true);
  const [isSearchLoading, setIsSearchLoading] = useState(false);
  const [isExplanationLoading, setIsExplanationLoading] = useState(false);
  const deferredQuery = useDeferredValue(query);

  useEffect(() => {
    let cancelled = false;

    async function loadReadiness() {
      try {
        const payload = await getJson<HealthResponse>("/health/ready");
        if (!cancelled) {
          setReadiness(payload.status);
          if (payload.snapshot) {
            setSnapshot(payload.snapshot);
          }
        }
      } catch (error) {
        if (!cancelled) {
          setReadiness("unavailable");
          setStatus(error instanceof Error ? error.message : "Unable to reach API health endpoint.");
        }
      }
    }

    loadReadiness();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadRecommendations() {
      try {
        setIsFeedLoading(true);
        setStatus("Loading personalized recommendations...");

        const payload = await getJson<RecommendationResponse>(
          `/recommend/${encodeURIComponent(customerId)}?k=8&mode=${encodeURIComponent(mode)}`
        );

        if (!cancelled) {
          startTransition(() => {
            setRecommendations(payload.recommendations);
            setSnapshot(payload.meta.snapshot);
            setStatus(`Loaded ${payload.recommendations.length} recommendations for ${payload.customer_id}.`);
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
          `/search?q=${encodeURIComponent(deferredQuery)}&k=8`
        );

        if (!cancelled) {
          startTransition(() => {
            setSearchResults(payload.results);
            setSnapshot(payload.meta.snapshot);
          });
        }
      } catch (error) {
        if (!cancelled) {
          setStatus(error instanceof Error ? error.message : "Failed to search the catalog.");
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
    if (!selectedItem) {
      setExplanation(null);
      return;
    }

    const activeItem = selectedItem;
    let cancelled = false;

    async function loadExplanation() {
      try {
        setIsExplanationLoading(true);
        setStatus(`Loading explanation for ${getItemText(activeItem, "prod_name", activeItem.article_id)}...`);

        const payload = await getJson<ExplainResponse>(
          `/explain/${encodeURIComponent(customerId)}/${encodeURIComponent(activeItem.article_id)}`
        );

        if (!cancelled) {
          startTransition(() => {
            setExplanation(payload);
            setSnapshot(payload.meta.snapshot);
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

  const activeItem = selectedItem ?? recommendations[0] ?? searchResults[0] ?? null;
  const activeName = activeItem ? getItemText(activeItem, "prod_name", activeItem.article_id) : "Select an item";
  const activeDescription = activeItem
    ? getItemText(activeItem, "detail_desc", "No detail description is available for this article.")
    : "Choose a recommendation or search result to inspect the item.";
  const activeCategory = activeItem ? getItemText(activeItem, "product_group_name", "Fashion item") : "Fashion item";
  const activeColor = activeItem ? getItemText(activeItem, "colour_group_name", "Neutral") : "Neutral";
  const activeDepartment = activeItem ? getItemText(activeItem, "department_name", "H&M catalog") : "H&M catalog";
  const readinessLabel = readiness === "ready" ? "Production-ready artifacts" : readiness === "degraded" ? "Running with fallbacks" : "Checking service";
  const searchEmptyText = deferredQuery.trim()
    ? "No items matched that prompt. Try a product type, color, or occasion."
    : "Enter a style prompt or tap a preset to search the H&M catalog.";

  return (
    <main className="workspace-shell">
      <header className="topbar frame">
        <div className="topbar__title">
          <p className="topbar__eyebrow">Fashion recommender workspace</p>
          <h1>H&M catalog recommendations</h1>
          <p className="topbar__copy">
            Real product metadata, optional Kaggle image support, semantic search, collaborative ranking, and item-level explanations in one operator-facing surface.
          </p>
        </div>

        <div className="status-cluster">
          <span className="status-pill">{readinessLabel}</span>
          <span className="status-pill">{getFeedLabel(snapshot)}</span>
          <span className="status-pill">{getImageStatus(snapshot)}</span>
          <a className="status-pill status-pill--link" href={`${API_BASE_URL}/docs`} target="_blank" rel="noreferrer">
            API docs
          </a>
        </div>
      </header>

      <div className="app-grid frame">
        <aside className="rail">
          <section className="rail-section">
            <div className="rail-section__header">
              <p className="rail-section__eyebrow">Session controls</p>
              <h2>Request setup</h2>
            </div>

            <label className="field">
              <span>Customer ID</span>
              <input value={customerId} onChange={(event) => setCustomerId(event.target.value)} />
            </label>

            <div className="field">
              <span>Ranking mode</span>
              <div className="mode-list" role="group" aria-label="Ranking mode">
                {(
                  Object.entries(MODE_DETAILS) as Array<
                    [RecommendationMode, { label: string; copy: string }]
                  >
                ).map(([modeKey, modeDetail]) => (
                  <button
                    key={modeKey}
                    type="button"
                    className={`mode-option${mode === modeKey ? " is-active" : ""}`}
                    onClick={() => setMode(modeKey)}
                    aria-pressed={mode === modeKey}
                  >
                    <strong>{modeDetail.label}</strong>
                    <span>{modeDetail.copy}</span>
                  </button>
                ))}
              </div>
            </div>

            <label className="field">
              <span>Semantic query</span>
              <textarea
                rows={4}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Examples: black summer dress, tailored office trousers, oversized knit cardigan"
              />
            </label>

            <div className="field">
              <span>Quick prompts</span>
              <div className="quick-list">
                {QUICK_QUERIES.map((quickQuery) => (
                  <button
                    key={quickQuery}
                    type="button"
                    className={`quick-option${query === quickQuery ? " is-active" : ""}`}
                    onClick={() => setQuery(quickQuery)}
                  >
                    {quickQuery}
                  </button>
                ))}
              </div>
            </div>
          </section>

          <section className="rail-section">
            <div className="rail-section__header">
              <p className="rail-section__eyebrow">Dataset state</p>
              <h2>Current catalog</h2>
            </div>

            <div className="metric-list">
              <div className="metric-row">
                <span>Articles</span>
                <strong>{formatNumber(snapshot?.catalog.article_count)}</strong>
              </div>
              <div className="metric-row">
                <span>Transactions</span>
                <strong>{formatNumber(snapshot?.catalog.interaction_count)}</strong>
              </div>
              <div className="metric-row">
                <span>Customers</span>
                <strong>{formatNumber(snapshot?.catalog.customer_count)}</strong>
              </div>
              <div className="metric-row">
                <span>Local images</span>
                <strong>{formatNumber(snapshot?.catalog.image_count)}</strong>
              </div>
            </div>

            <p className="helper-copy">
              Put the Kaggle image files under <code>data/raw/images</code> and rerun the article prep step to populate image metadata in the cleaned catalog.
            </p>
          </section>

          <section className="rail-section rail-section--compact">
            <div className="rail-section__header">
              <p className="rail-section__eyebrow">Runtime</p>
              <h2>Live status</h2>
            </div>
            <p className="status-line">{status}</p>
          </section>
        </aside>

        <div className="main-column">
          <SectionShell
            title="Recommendation feed"
            subtitle={`${MODE_DETAILS[mode].label} ranking`}
            meta={isFeedLoading ? "Loading" : `${recommendations.length} items`}
          >
            <div className="catalog-grid">
              {recommendations.map((item) => (
                <ProductCard
                  key={`rec-${item.article_id}`}
                  item={item}
                  apiBaseUrl={API_BASE_URL}
                  label="Recommended"
                  selected={activeItem?.article_id === item.article_id}
                  onSelect={setSelectedItem}
                />
              ))}
            </div>

            {recommendations.length === 0 ? (
              <p className="empty-state">
                {isFeedLoading ? "Loading recommendation feed..." : "No recommendations returned for this customer."}
              </p>
            ) : null}
          </SectionShell>

          <SectionShell
            title="Semantic search"
            subtitle="Natural-language retrieval"
            meta={isSearchLoading ? "Searching" : `${searchResults.length} matches`}
          >
            <div className="catalog-grid catalog-grid--search">
              {searchResults.map((item) => (
                <ProductCard
                  key={`search-${item.article_id}`}
                  item={item}
                  apiBaseUrl={API_BASE_URL}
                  label="Search match"
                  selected={activeItem?.article_id === item.article_id}
                  onSelect={setSelectedItem}
                />
              ))}
            </div>

            {searchResults.length === 0 ? <p className="empty-state">{searchEmptyText}</p> : null}
          </SectionShell>

          <section className="system-panel">
            <div className="system-panel__header">
              <div>
                <p className="panel-section__subtitle">System snapshot</p>
                <h2>Model and artifact readiness</h2>
              </div>
            </div>

            <div className="readiness-grid">
              <div className="readiness-item">
                <span>Graph engine</span>
                <strong>{snapshot?.engines.graph_ready ? "Ready" : "Fallback"}</strong>
              </div>
              <div className="readiness-item">
                <span>Semantic engine</span>
                <strong>{snapshot?.engines.semantic_ready ? "Ready" : "Fallback"}</strong>
              </div>
              <div className="readiness-item">
                <span>Reranker</span>
                <strong>{snapshot?.engines.reranker_ready ? "Ready" : "Missing"}</strong>
              </div>
              <div className="readiness-item">
                <span>Images directory</span>
                <strong>{snapshot?.artifacts.images_ready ? "Available" : "Missing"}</strong>
              </div>
            </div>
          </section>
        </div>

        <aside className="inspector">
          <section className="inspector__section">
            <div className="inspector__header">
              <p className="rail-section__eyebrow">Selected item</p>
              <h2>{activeName}</h2>
            </div>

            {activeItem ? (
              <ProductMedia
                item={activeItem}
                apiBaseUrl={API_BASE_URL}
                className="inspector__media"
                alt={activeName}
              >
                <span className="inspector__source">{activeItem.source}</span>
              </ProductMedia>
            ) : (
              <div className="inspector__placeholder">Select an item to inspect the catalog entry.</div>
            )}

            <p className="inspector__description">{activeDescription}</p>

            <div className="chip-row">
              <span className="chip">{activeCategory}</span>
              <span className="chip">{activeColor}</span>
              <span className="chip">{activeDepartment}</span>
            </div>

            <div className="metric-list">
              <div className="metric-row">
                <span>Article ID</span>
                <strong>{activeItem?.article_id ?? "-"}</strong>
              </div>
              <div className="metric-row">
                <span>Source</span>
                <strong>{activeItem?.source ?? "-"}</strong>
              </div>
              <div className="metric-row">
                <span>Score</span>
                <strong>{activeItem ? activeItem.score.toFixed(3) : "-"}</strong>
              </div>
              <div className="metric-row">
                <span>Color tone</span>
                <strong style={{ color: getProductTone(activeColor) }}>{activeColor}</strong>
              </div>
            </div>
          </section>

          <section className="inspector__section">
            <div className="inspector__header">
              <p className="rail-section__eyebrow">Explanation</p>
              <h2>Why this item</h2>
            </div>

            {isExplanationLoading ? (
              <p className="empty-state">Loading explanation...</p>
            ) : explanation ? (
              <ol className="reason-list">
                {explanation.reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ol>
            ) : (
              <p className="empty-state">Select a recommendation or search result to inspect the explanation output.</p>
            )}
          </section>
        </aside>
      </div>
    </main>
  );
}
