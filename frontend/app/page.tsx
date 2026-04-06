"use client";

import { startTransition, useDeferredValue, useEffect, useState } from "react";

import {
  ProductCard,
  type RecommendationItem,
  getItemBoolean,
  getItemImageUrl,
  getItemText
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

type SearchResponse = {
  query: string;
  results: RecommendationItem[];
  meta: ResponseMeta;
};

type DiscoverResponse = {
  query: string;
  anchor: RecommendationItem | null;
  mode: RecommendationMode;
  recommendations: RecommendationItem[];
  meta: ResponseMeta;
};

type RelatedResponse = {
  anchor_article_id: string;
  anchor: RecommendationItem | null;
  mode: RecommendationMode;
  recommendations: RecommendationItem[];
  meta: ResponseMeta;
};

type RelatedExplainResponse = {
  anchor_article_id: string;
  article_id: string;
  reasons: string[];
  meta: ResponseMeta;
};

type HealthResponse = {
  status: string;
  snapshot?: ServiceSnapshot | null;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

const QUICK_QUERIES = [
  "black summer dress",
  "wide leg jeans",
  "white shirt",
  "sports bra",
  "oversized hoodie"
] as const;

const MODE_DETAILS: Record<RecommendationMode, string> = {
  hybrid: "Blend graph, semantic, and co-purchase signals",
  gnn: "Prioritize graph and behavior similarity",
  semantic: "Prioritize item text similarity"
};

const JOURNEY_STEPS = [
  "Write the product or phrase you have in mind.",
  "Pick the closest anchor item from the catalog.",
  "Inspect the ranked set and why each item surfaced."
] as const;

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

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="detail-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SystemRow({
  label,
  value,
  tone = "neutral"
}: {
  label: string;
  value: string;
  tone?: "neutral" | "good" | "warn";
}) {
  return (
    <div className={`system-row system-row--${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ProductStage({
  item,
  title,
  subtitle
}: {
  item: RecommendationItem | null;
  title: string;
  subtitle: string;
}) {
  const name = item ? getItemText(item, "prod_name", item.article_id) : "No product selected";
  const category = item ? getItemText(item, "product_group_name", "Fashion item") : "Awaiting selection";
  const color = item ? getItemText(item, "colour_group_name", "Unknown") : "No color";
  const description = item
    ? getItemText(item, "detail_desc", "No detail description is available.")
    : "Search for a product first. The system will pin the closest catalog item here and use it as the anchor for recommendations.";
  const imageUrl = item ? getItemImageUrl(item, API_BASE_URL) : null;
  const imageReady = item ? getItemBoolean(item, "image_available") : false;

  return (
    <section className="stage-block">
      <div className="stage-block__header">
        <p className="kicker">{subtitle}</p>
        <h2>{title}</h2>
      </div>

      <div className="stage-visual">
        {imageUrl ? (
          <img src={imageUrl} alt={name} />
        ) : (
          <div className="stage-visual__fallback">
            <span>{name.slice(0, 1).toUpperCase() || "F"}</span>
            <small>{imageReady ? "Image loading" : "Image archive not connected"}</small>
          </div>
        )}
      </div>

      <div className="stage-copy">
        <h3>{name}</h3>
        <p>{description}</p>
      </div>

      <div className="detail-list">
        <DetailRow label="Category" value={category} />
        <DetailRow label="Color" value={color} />
        <DetailRow label="Article ID" value={item?.article_id ?? "-"} />
        <DetailRow label="Image state" value={imageReady ? "Connected" : "Unavailable"} />
      </div>
    </section>
  );
}

export default function Page() {
  const [query, setQuery] = useState("black summer dress");
  const [mode, setMode] = useState<RecommendationMode>("hybrid");
  const [searchResults, setSearchResults] = useState<RecommendationItem[]>([]);
  const [anchorItem, setAnchorItem] = useState<RecommendationItem | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([]);
  const [selectedRecommendation, setSelectedRecommendation] = useState<RecommendationItem | null>(null);
  const [reasons, setReasons] = useState<string[]>([]);
  const [snapshot, setSnapshot] = useState<ServiceSnapshot | null>(null);
  const [status, setStatus] = useState("Search for a product to let the recommender build a ranked set.");
  const [isLoadingSearch, setIsLoadingSearch] = useState(false);
  const [isLoadingRecommendations, setIsLoadingRecommendations] = useState(false);
  const [isLoadingReasons, setIsLoadingReasons] = useState(false);
  const deferredQuery = useDeferredValue(query);

  useEffect(() => {
    let cancelled = false;

    async function loadHealth() {
      try {
        const payload = await getJson<HealthResponse>("/health/ready");
        if (!cancelled && payload.snapshot) {
          setSnapshot(payload.snapshot);
        }
      } catch {
        if (!cancelled) {
          setStatus("API health check failed. Make sure the backend is running.");
        }
      }
    }

    loadHealth();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!deferredQuery.trim()) {
      setSearchResults([]);
      setAnchorItem(null);
      setRecommendations([]);
      setSelectedRecommendation(null);
      setReasons([]);
      return;
    }

    let cancelled = false;

    async function loadDiscovery() {
      try {
        setIsLoadingSearch(true);
        setIsLoadingRecommendations(true);
        setStatus("Searching the catalog and assembling the related set...");

        const [searchPayload, discoverPayload] = await Promise.all([
          getJson<SearchResponse>(`/search?q=${encodeURIComponent(deferredQuery)}&k=8`),
          getJson<DiscoverResponse>(
            `/discover?q=${encodeURIComponent(deferredQuery)}&k=8&mode=${encodeURIComponent(mode)}`
          )
        ]);

        if (!cancelled) {
          startTransition(() => {
            setSearchResults(searchPayload.results);
            setAnchorItem(discoverPayload.anchor);
            setRecommendations(discoverPayload.recommendations);
            setSelectedRecommendation(discoverPayload.recommendations[0] ?? null);
            setReasons([]);
            setSnapshot(discoverPayload.meta.snapshot);
            setStatus(
              discoverPayload.anchor
                ? `Anchor locked to ${getItemText(discoverPayload.anchor, "prod_name", discoverPayload.anchor.article_id)}.`
                : "No anchor product found for that query."
            );
          });
        }
      } catch (error) {
        if (!cancelled) {
          setStatus(error instanceof Error ? error.message : "Discovery request failed.");
        }
      } finally {
        if (!cancelled) {
          setIsLoadingSearch(false);
          setIsLoadingRecommendations(false);
        }
      }
    }

    loadDiscovery();
    return () => {
      cancelled = true;
    };
  }, [deferredQuery, mode]);

  useEffect(() => {
    if (!anchorItem || !selectedRecommendation) {
      setReasons([]);
      return;
    }

    const activeAnchor = anchorItem;
    const activeRecommendation = selectedRecommendation;
    let cancelled = false;

    async function loadReasons() {
      try {
        setIsLoadingReasons(true);
        const payload = await getJson<RelatedExplainResponse>(
          `/explain-related/${encodeURIComponent(activeAnchor.article_id)}/${encodeURIComponent(activeRecommendation.article_id)}`
        );

        if (!cancelled) {
          startTransition(() => {
            setReasons(payload.reasons);
            setSnapshot(payload.meta.snapshot);
          });
        }
      } catch (error) {
        if (!cancelled) {
          setStatus(error instanceof Error ? error.message : "Failed to explain recommendation.");
        }
      } finally {
        if (!cancelled) {
          setIsLoadingReasons(false);
        }
      }
    }

    loadReasons();
    return () => {
      cancelled = true;
    };
  }, [anchorItem, selectedRecommendation]);

  async function loadRelatedForItem(item: RecommendationItem) {
    try {
      setIsLoadingRecommendations(true);
      setStatus(`Building a related set from ${getItemText(item, "prod_name", item.article_id)}...`);

      const payload = await getJson<RelatedResponse>(
        `/related/${encodeURIComponent(item.article_id)}?k=8&mode=${encodeURIComponent(mode)}`
      );

      startTransition(() => {
        setAnchorItem(payload.anchor ?? item);
        setRecommendations(payload.recommendations);
        setSelectedRecommendation(payload.recommendations[0] ?? null);
        setReasons([]);
        setSnapshot(payload.meta.snapshot);
        setStatus(`Loaded ${payload.recommendations.length} related products from the selected anchor.`);
      });
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Failed to load related products.");
    } finally {
      setIsLoadingRecommendations(false);
    }
  }

  const recommendationTitle = selectedRecommendation
    ? getItemText(selectedRecommendation, "prod_name", selectedRecommendation.article_id)
    : "No recommendation selected";

  return (
    <main className="atelier-app">
      <section className="hero-band frame">
        <div className="hero-band__copy">
          <p className="kicker">Fashion recommender workspace</p>
          <h1>Search one product and let the model expand the rack around it.</h1>
          <p className="lead">
            Write a product name, color, or intent like <code>black summer dress</code>. The app finds the nearest
            catalog anchor, ranks similar items through the recommendation stack, and explains why each result belongs
            in the set.
          </p>
        </div>

        <div className="hero-band__stats">
          <span className="hero-pill">{formatNumber(snapshot?.catalog.article_count)} articles</span>
          <span className="hero-pill">{formatNumber(snapshot?.catalog.customer_count)} customers</span>
          <span className="hero-pill">{snapshot?.engines.fallback_active ? "Fallback mode" : "Model mode"}</span>
          <span className="hero-pill">{MODE_DETAILS[mode]}</span>
        </div>
      </section>

      <section className="journey-strip frame" aria-label="How the workspace works">
        {JOURNEY_STEPS.map((step, index) => (
          <div key={step} className="journey-step">
            <span>{String(index + 1).padStart(2, "0")}</span>
            <p>{step}</p>
          </div>
        ))}
      </section>

      <div className="atelier-grid frame">
        <aside className="control-rail">
          <section className="surface surface--dense">
            <div className="surface__header">
              <div>
                <p className="kicker">Search brief</p>
                <h2>Tell the system what you want</h2>
              </div>
            </div>

            <label className="search-label" htmlFor="product-query">
              Product or phrase
            </label>
            <input
              id="product-query"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="black summer dress, cropped denim jacket, white shirt"
            />

            <div className="mode-switcher" role="group" aria-label="Recommendation mode">
              {(Object.keys(MODE_DETAILS) as RecommendationMode[]).map((modeKey) => (
                <button
                  key={modeKey}
                  type="button"
                  className={`mode-button${mode === modeKey ? " is-active" : ""}`}
                  onClick={() => setMode(modeKey)}
                >
                  <span>{modeKey}</span>
                  <small>{MODE_DETAILS[modeKey]}</small>
                </button>
              ))}
            </div>

            <div className="status-callout">
              <p className="kicker">Live status</p>
              <p>{status}</p>
            </div>
          </section>

          <section className="surface surface--dense">
            <div className="surface__header">
              <div>
                <p className="kicker">Quick prompts</p>
                <h2>Jump into the catalog</h2>
              </div>
            </div>

            <div className="quick-actions">
              {QUICK_QUERIES.map((quickQuery) => (
                <button key={quickQuery} type="button" className="quick-chip" onClick={() => setQuery(quickQuery)}>
                  {quickQuery}
                </button>
              ))}
            </div>
          </section>

          <section className="surface surface--dense">
            <div className="surface__header">
              <div>
                <p className="kicker">System readiness</p>
                <h2>What is live in the stack</h2>
              </div>
            </div>

            <div className="system-list">
              <SystemRow
                label="Graph engine"
                value={snapshot?.engines.graph_ready ? "Ready" : "Missing"}
                tone={snapshot?.engines.graph_ready ? "good" : "warn"}
              />
              <SystemRow
                label="Semantic engine"
                value={snapshot?.engines.semantic_ready ? "Ready" : "Missing"}
                tone={snapshot?.engines.semantic_ready ? "good" : "warn"}
              />
              <SystemRow
                label="Image archive"
                value={snapshot?.artifacts.images_ready ? "Connected" : "Not loaded"}
                tone={snapshot?.artifacts.images_ready ? "good" : "warn"}
              />
              <SystemRow
                label="Catalog photos"
                value={formatNumber(snapshot?.catalog.image_count)}
                tone={snapshot?.catalog.image_count ? "good" : "warn"}
              />
              <SystemRow
                label="Interactions"
                value={formatNumber(snapshot?.catalog.interaction_count)}
              />
            </div>
          </section>
        </aside>

        <section className="workspace-column">
          <SectionShell
            title="Search Matches"
            subtitle="Closest catalog anchors"
            meta={isLoadingSearch ? "Searching" : `${searchResults.length} results`}
          >
            <div className="product-list">
              {searchResults.map((item) => (
                <ProductCard
                  key={`search-${item.article_id}`}
                  item={item}
                  label="Anchor option"
                  imageBaseUrl={API_BASE_URL}
                  selected={anchorItem?.article_id === item.article_id}
                  onSelect={loadRelatedForItem}
                />
              ))}
            </div>
            {searchResults.length === 0 ? <p className="empty-copy">Closest catalog matches will appear here.</p> : null}
          </SectionShell>

          <SectionShell
            title="Recommended Set"
            subtitle="Items ranked from the selected anchor"
            meta={isLoadingRecommendations ? "Loading" : `${recommendations.length} items`}
          >
            <div className="product-list">
              {recommendations.map((item) => (
                <ProductCard
                  key={`rec-${item.article_id}`}
                  item={item}
                  label="Recommended"
                  imageBaseUrl={API_BASE_URL}
                  selected={selectedRecommendation?.article_id === item.article_id}
                  onSelect={setSelectedRecommendation}
                />
              ))}
            </div>
            {recommendations.length === 0 ? (
              <p className="empty-copy">Recommendations will appear here after the anchor product is set.</p>
            ) : null}
          </SectionShell>
        </section>

        <aside className="spotlight-column">
          <section className="spotlight surface">
            <ProductStage item={anchorItem} title="Anchor product" subtitle="Matched from your search" />

            <div className="spotlight-divider" />

            <ProductStage item={selectedRecommendation} title="Selected recommendation" subtitle="Current focus" />

            <div className="spotlight-divider" />

            <section className="reason-block">
              <div className="surface__header">
                <div>
                  <p className="kicker">Recommendation logic</p>
                  <h2>Why {recommendationTitle} surfaced</h2>
                </div>
              </div>

              {isLoadingReasons ? (
                <p className="empty-copy">Explaining the recommendation...</p>
              ) : reasons.length > 0 ? (
                <ol className="reason-list">
                  {reasons.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ol>
              ) : (
                <p className="empty-copy">Select a recommendation to see the model explanation.</p>
              )}
            </section>
          </section>
        </aside>
      </div>
    </main>
  );
}
