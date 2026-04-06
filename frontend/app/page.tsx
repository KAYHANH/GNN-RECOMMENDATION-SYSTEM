"use client";

import { startTransition, useEffect, useRef, useState, type FormEvent } from "react";

import {
  ProductCard,
  type RecommendationItem,
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
  hybrid: "Blend graph behavior, text similarity, and co-purchase signals.",
  gnn: "Bias toward graph and behavioral similarity.",
  semantic: "Bias toward product language and metadata similarity."
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

function RunwayProduct({
  title,
  item
}: {
  title: string;
  item: RecommendationItem | null;
}) {
  const imageUrl = item ? getItemImageUrl(item, API_BASE_URL) : null;
  const name = item ? getItemText(item, "prod_name", item.article_id) : "Waiting for a product";
  const category = item ? getItemText(item, "product_group_name", "Fashion item") : "No category";
  const color = item ? getItemText(item, "colour_group_name", "Unknown") : "No color";
  const description = item
    ? getItemText(item, "detail_desc", "No detail description is available.")
    : "Submit a search and pick an anchor product to populate the runway.";

  return (
    <article className="runway-product">
      <div className="runway-product__media">
        {imageUrl ? (
          <img src={imageUrl} alt={name} />
        ) : (
          <div className="runway-product__fallback">
            <span>{name.charAt(0).toUpperCase() || "F"}</span>
            <small>Product image unavailable</small>
          </div>
        )}
        <div className="runway-product__overlay">
          <p>{title}</p>
          <strong>{category}</strong>
        </div>
      </div>

      <div className="runway-product__copy">
        <h2>{name}</h2>
        <p>{description}</p>
        <div className="runway-product__facts">
          <span>{color}</span>
          <span>{item?.article_id ?? "-"}</span>
          <span>{item ? item.score.toFixed(3) : "-"}</span>
        </div>
      </div>
    </article>
  );
}

export default function Page() {
  const [draftQuery, setDraftQuery] = useState("black summer dress");
  const [activeQuery, setActiveQuery] = useState("black summer dress");
  const [mode, setMode] = useState<RecommendationMode>("hybrid");
  const [searchResults, setSearchResults] = useState<RecommendationItem[]>([]);
  const [anchorItem, setAnchorItem] = useState<RecommendationItem | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([]);
  const [selectedRecommendation, setSelectedRecommendation] = useState<RecommendationItem | null>(null);
  const [reasons, setReasons] = useState<string[]>([]);
  const [snapshot, setSnapshot] = useState<ServiceSnapshot | null>(null);
  const [status, setStatus] = useState("Search for a product and lock an anchor item from the H&M catalog.");
  const [isSearching, setIsSearching] = useState(false);
  const [isLoadingRelated, setIsLoadingRelated] = useState(false);
  const [isLoadingReasons, setIsLoadingReasons] = useState(false);

  const discoveryRequestRef = useRef(0);
  const reasonRequestRef = useRef(0);

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
    if (!activeQuery.trim()) {
      return;
    }

    const requestId = ++discoveryRequestRef.current;

    async function loadDiscovery() {
      try {
        setIsSearching(true);
        setStatus(`Searching the catalog for "${activeQuery}"...`);

        const [searchPayload, discoverPayload] = await Promise.all([
          getJson<SearchResponse>(`/search?q=${encodeURIComponent(activeQuery)}&k=8`),
          getJson<DiscoverResponse>(`/discover?q=${encodeURIComponent(activeQuery)}&k=8&mode=${encodeURIComponent(mode)}`)
        ]);

        if (requestId !== discoveryRequestRef.current) {
          return;
        }

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
              : `No anchor product found for "${activeQuery}".`
          );
        });
      } catch (error) {
        if (requestId === discoveryRequestRef.current) {
          setStatus(error instanceof Error ? error.message : "Discovery request failed.");
        }
      } finally {
        if (requestId === discoveryRequestRef.current) {
          setIsSearching(false);
        }
      }
    }

    loadDiscovery();
  }, [activeQuery, mode]);

  useEffect(() => {
    if (!anchorItem || !selectedRecommendation) {
      setReasons([]);
      return;
    }

    const activeAnchor = anchorItem;
    const activeRecommendation = selectedRecommendation;
    const requestId = ++reasonRequestRef.current;

    async function loadReasons() {
      try {
        setIsLoadingReasons(true);
        const payload = await getJson<RelatedExplainResponse>(
          `/explain-related/${encodeURIComponent(activeAnchor.article_id)}/${encodeURIComponent(activeRecommendation.article_id)}`
        );

        if (requestId !== reasonRequestRef.current) {
          return;
        }

        startTransition(() => {
          setReasons(payload.reasons);
          setSnapshot(payload.meta.snapshot);
        });
      } catch (error) {
        if (requestId === reasonRequestRef.current) {
          setStatus(error instanceof Error ? error.message : "Failed to explain the recommendation.");
        }
      } finally {
        if (requestId === reasonRequestRef.current) {
          setIsLoadingReasons(false);
        }
      }
    }

    loadReasons();
  }, [anchorItem, selectedRecommendation]);

  async function handleSearchSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextQuery = draftQuery.trim();
    if (!nextQuery) {
      setStatus("Write a product name or phrase first.");
      return;
    }
    setActiveQuery(nextQuery);
  }

  async function handleAnchorPick(item: RecommendationItem) {
    try {
      setIsLoadingRelated(true);
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
        setStatus(`Loaded ${payload.recommendations.length} recommendations from the selected anchor.`);
      });
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Failed to load related products.");
    } finally {
      setIsLoadingRelated(false);
    }
  }

  function applyQuickQuery(query: string) {
    setDraftQuery(query);
    setActiveQuery(query);
  }

  const selectedName = selectedRecommendation
    ? getItemText(selectedRecommendation, "prod_name", selectedRecommendation.article_id)
    : "your selected recommendation";

  return (
    <main className="studio-page">
      <header className="command-deck">
        <div className="command-deck__intro">
          <p className="eyebrow">Fashion recommendation studio</p>
          <h1>Search the product you want and let the model build the rack around it.</h1>
          <p className="command-deck__copy">
            This workspace is built for one job: write a product phrase, choose the closest H&M item, and inspect
            the products your recommendation stack thinks belong next to it.
          </p>
        </div>

        <form className="search-console" onSubmit={handleSearchSubmit}>
          <label htmlFor="search-query">Product search</label>
          <div className="search-console__row">
            <input
              id="search-query"
              value={draftQuery}
              onChange={(event) => setDraftQuery(event.target.value)}
              placeholder="black summer dress, cropped jacket, white shirt"
            />
            <button type="submit" className="search-submit" disabled={isSearching}>
              {isSearching ? "Searching" : "Search"}
            </button>
          </div>

          <div className="mode-pills" role="group" aria-label="Recommendation mode">
            {(Object.keys(MODE_DETAILS) as RecommendationMode[]).map((modeKey) => (
              <button
                key={modeKey}
                type="button"
                className={`mode-pill${mode === modeKey ? " is-active" : ""}`}
                onClick={() => setMode(modeKey)}
              >
                {modeKey}
              </button>
            ))}
          </div>

          <div className="quick-prompt-row">
            {QUICK_QUERIES.map((quickQuery) => (
              <button key={quickQuery} type="button" className="quick-prompt" onClick={() => applyQuickQuery(quickQuery)}>
                {quickQuery}
              </button>
            ))}
          </div>
        </form>
      </header>

      <section className="status-ribbon">
        <span>{status}</span>
        <div className="status-ribbon__stats">
          <strong>{formatNumber(snapshot?.catalog.article_count)} products</strong>
          <strong>{formatNumber(snapshot?.catalog.image_count)} images</strong>
          <strong>{snapshot?.engines.fallback_active ? "fallback" : "model"}</strong>
        </div>
      </section>

      <section className="runway-shell">
        <aside className="runway-shell__left">
          <SectionShell
            title="Anchor Options"
            subtitle="Closest search matches"
            meta={isSearching ? "Searching" : `${searchResults.length} matches`}
          >
            <div className="product-list">
              {searchResults.map((item) => (
                <ProductCard
                  key={`match-${item.article_id}`}
                  item={item}
                  label="Anchor"
                  imageBaseUrl={API_BASE_URL}
                  selected={anchorItem?.article_id === item.article_id}
                  onSelect={handleAnchorPick}
                />
              ))}
            </div>
          </SectionShell>
        </aside>

        <section className="runway-shell__center">
          <div className="runway-stage">
            <RunwayProduct title="Anchor product" item={anchorItem} />
            <RunwayProduct title="Selected recommendation" item={selectedRecommendation} />
          </div>

          <div className="logic-strip">
            <div className="logic-strip__intro">
              <p className="eyebrow">Why it surfaced</p>
              <h2>{selectedName}</h2>
              <p>{MODE_DETAILS[mode]}</p>
            </div>

            {isLoadingReasons ? (
              <p className="logic-strip__empty">Explaining the selected recommendation...</p>
            ) : reasons.length > 0 ? (
              <ol className="logic-list">
                {reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ol>
            ) : (
              <p className="logic-strip__empty">Pick a recommendation to inspect the reasoning.</p>
            )}
          </div>
        </section>

        <aside className="runway-shell__right">
          <SectionShell
            title="Recommended Rack"
            subtitle="Model-ranked items"
            meta={isLoadingRelated ? "Refreshing" : `${recommendations.length} items`}
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
          </SectionShell>
        </aside>
      </section>

      <footer className="ops-band">
        <div className="ops-band__block">
          <p className="eyebrow">System state</p>
          <h2>What is live right now</h2>
        </div>

        <div className="ops-band__metrics">
          <div>
            <span>Graph engine</span>
            <strong>{snapshot?.engines.graph_ready ? "Ready" : "Missing"}</strong>
          </div>
          <div>
            <span>Semantic engine</span>
            <strong>{snapshot?.engines.semantic_ready ? "Ready" : "Missing"}</strong>
          </div>
          <div>
            <span>Public image feed</span>
            <strong>{snapshot?.catalog.image_count ? "Connected" : "Unavailable"}</strong>
          </div>
          <div>
            <span>Interactions</span>
            <strong>{formatNumber(snapshot?.catalog.interaction_count)}</strong>
          </div>
        </div>
      </footer>
    </main>
  );
}
