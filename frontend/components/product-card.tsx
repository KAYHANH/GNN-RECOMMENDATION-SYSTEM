import type { CSSProperties } from "react";

export type RecommendationItem = {
  article_id: string;
  score: number;
  source: string;
  metadata: Record<string, unknown>;
};

type ProductCardProps = {
  item: RecommendationItem;
  eyebrow: string;
  actionLabel: string;
  onAction: (item: RecommendationItem) => void;
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
    return "#4b342d";
  }

  if (normalized.includes("white") || normalized.includes("cream") || normalized.includes("beige")) {
    return "#a38c75";
  }

  if (normalized.includes("blue")) {
    return "#526c8d";
  }

  if (normalized.includes("red") || normalized.includes("burgundy")) {
    return "#8c493f";
  }

  if (normalized.includes("green") || normalized.includes("olive")) {
    return "#667058";
  }

  if (normalized.includes("pink") || normalized.includes("rose")) {
    return "#a56d76";
  }

  return "#7c6558";
}

function shrinkText(text: string, maxLength: number): string {
  if (text.length <= maxLength) {
    return text;
  }

  return `${text.slice(0, maxLength - 3).trimEnd()}...`;
}

export function ProductCard({
  item,
  eyebrow,
  actionLabel,
  onAction,
  selected = false
}: ProductCardProps) {
  const productName = getItemText(item, "prod_name", item.article_id);
  const color = getItemText(item, "colour_group_name", "Unknown color");
  const category = getItemText(item, "product_group_name", "Fashion item");
  const description = getItemText(item, "detail_desc", "No long description available yet for this item.");
  const visualStyle: CSSProperties = {
    backgroundImage: `linear-gradient(140deg, ${getProductTone(color)} 0%, rgba(255, 255, 255, 0.16) 38%, rgba(18, 15, 13, 0.46) 100%)`
  };

  return (
    <button
      type="button"
      className={`product-tile${selected ? " is-selected" : ""}`}
      onClick={() => onAction(item)}
    >
      <span className="product-tile__visual" style={visualStyle}>
        <span className="product-tile__label">{category}</span>
        <span className="product-tile__id">#{item.article_id.slice(-4)}</span>
      </span>

      <span className="product-tile__body">
        <span className="product-tile__eyebrow">
          <span>{eyebrow}</span>
          <span>{item.source}</span>
        </span>

        <strong className="product-tile__title">{productName}</strong>
        <span className="product-tile__desc">{shrinkText(description, 140)}</span>

        <span className="product-tile__footer">
          <span>{color}</span>
          <span>
            {actionLabel} - {item.score.toFixed(3)}
          </span>
        </span>
      </span>
    </button>
  );
}
