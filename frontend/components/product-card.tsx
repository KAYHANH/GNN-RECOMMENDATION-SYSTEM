import type { CSSProperties, ReactNode } from "react";
import { useEffect, useState } from "react";

export type RecommendationItem = {
  article_id: string;
  score: number;
  source: string;
  metadata: Record<string, unknown>;
};

type ProductMediaProps = {
  item: RecommendationItem;
  apiBaseUrl: string;
  className: string;
  alt: string;
  children?: ReactNode;
};

type ProductCardProps = {
  item: RecommendationItem;
  apiBaseUrl: string;
  label: string;
  onSelect: (item: RecommendationItem) => void;
  selected?: boolean;
};

export function getItemText(item: RecommendationItem, key: string, fallback: string): string {
  const value = item.metadata?.[key];

  if (typeof value === "string" && value.trim()) {
    return value;
  }

  if (typeof value === "number") {
    return String(value);
  }

  return fallback;
}

export function getProductTone(color: string): string {
  const normalized = color.toLowerCase();

  if (normalized.includes("black")) {
    return "#5f5149";
  }

  if (normalized.includes("white") || normalized.includes("cream") || normalized.includes("beige")) {
    return "#c7b29a";
  }

  if (normalized.includes("blue")) {
    return "#6e87a5";
  }

  if (normalized.includes("red") || normalized.includes("burgundy")) {
    return "#9d5a4d";
  }

  if (normalized.includes("green") || normalized.includes("olive")) {
    return "#7a8668";
  }

  if (normalized.includes("pink") || normalized.includes("rose")) {
    return "#b18086";
  }

  return "#8e796c";
}

export function getArticleImageUrl(item: RecommendationItem, apiBaseUrl: string): string {
  const metadataUrl = item.metadata?.image_url;
  if (typeof metadataUrl === "string" && metadataUrl.trim()) {
    return metadataUrl;
  }

  return `${apiBaseUrl}/catalog/images/${encodeURIComponent(item.article_id)}`;
}

function getFallbackStyle(item: RecommendationItem): CSSProperties {
  const color = getItemText(item, "colour_group_name", "neutral");
  const tone = getProductTone(color);

  return {
    backgroundImage: `linear-gradient(155deg, ${tone} 0%, rgba(255, 255, 255, 0.24) 42%, rgba(28, 24, 22, 0.78) 100%)`
  };
}

function shrinkText(text: string, maxLength: number): string {
  if (text.length <= maxLength) {
    return text;
  }

  return `${text.slice(0, maxLength - 3).trimEnd()}...`;
}

export function ProductMedia({ item, apiBaseUrl, className, alt, children }: ProductMediaProps) {
  const [imageFailed, setImageFailed] = useState(false);
  const imageUrl = getArticleImageUrl(item, apiBaseUrl);

  useEffect(() => {
    setImageFailed(false);
  }, [imageUrl, item.article_id]);

  return (
    <div className={className} style={getFallbackStyle(item)}>
      {!imageFailed ? (
        <img
          src={imageUrl}
          alt={alt}
          loading="lazy"
          onError={() => setImageFailed(true)}
        />
      ) : null}
      {children}
    </div>
  );
}

export function ProductCard({ item, apiBaseUrl, label, onSelect, selected = false }: ProductCardProps) {
  const productName = getItemText(item, "prod_name", item.article_id);
  const color = getItemText(item, "colour_group_name", "Unknown color");
  const category = getItemText(item, "product_group_name", "Fashion item");
  const description = getItemText(item, "detail_desc", "No detailed description is available yet.");

  return (
    <button
      type="button"
      className={`catalog-card${selected ? " is-selected" : ""}`}
      onClick={() => onSelect(item)}
    >
      <ProductMedia item={item} apiBaseUrl={apiBaseUrl} className="catalog-card__media" alt={productName}>
        <span className="catalog-card__badge">{label}</span>
      </ProductMedia>

      <div className="catalog-card__body">
        <div className="catalog-card__meta">
          <span>{category}</span>
          <span>{item.source}</span>
        </div>
        <h3>{productName}</h3>
        <p>{shrinkText(description, 108)}</p>
        <div className="catalog-card__footer">
          <span>{color}</span>
          <span>score {item.score.toFixed(3)}</span>
        </div>
      </div>
    </button>
  );
}
