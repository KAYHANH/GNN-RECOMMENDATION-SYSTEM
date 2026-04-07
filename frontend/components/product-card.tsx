"use client";

import { useEffect, useState } from "react";

export type RecommendationItem = {
  article_id: string;
  score: number;
  source: string;
  metadata: Record<string, unknown>;
};

type ProductCardProps = {
  item: RecommendationItem;
  label: string;
  imageBaseUrl: string;
  onSelect: (item: RecommendationItem) => void;
  selected?: boolean;
};

type ProductImageProps = {
  src: string | null;
  alt: string;
  fallbackLabel: string;
  variant: "card" | "runway";
  loading?: "lazy" | "eager";
};

const failedImageUrls = new Set<string>();

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

export function getItemBoolean(item: RecommendationItem, key: string): boolean {
  const value = item.metadata?.[key];
  return value === true || value === "true" || value === 1;
}

export function getItemImageUrl(item: RecommendationItem, imageBaseUrl: string): string | null {
  const directUrl = item.metadata?.image_external_url;
  if (typeof directUrl === "string" && directUrl.trim()) {
    return directUrl;
  }

  if (getItemBoolean(item, "image_available")) {
    return `${imageBaseUrl}/catalog/images/${encodeURIComponent(item.article_id)}`;
  }

  return null;
}

function shrinkText(text: string, maxLength: number): string {
  if (text.length <= maxLength) {
    return text;
  }

  return `${text.slice(0, maxLength - 3).trimEnd()}...`;
}

function getInitials(name: string): string {
  const initials = name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((chunk) => chunk[0]?.toUpperCase() ?? "")
    .join("");

  return initials || "HM";
}

export function ProductImage({
  src,
  alt,
  fallbackLabel,
  variant,
  loading = "lazy"
}: ProductImageProps) {
  const [hasFailed, setHasFailed] = useState(() => !src || (src ? failedImageUrls.has(src) : false));

  useEffect(() => {
    setHasFailed(!src || (src ? failedImageUrls.has(src) : false));
  }, [src]);

  if (!src || hasFailed) {
    if (variant === "runway") {
      return (
        <div className="runway-product__fallback">
          <span>{getInitials(fallbackLabel)}</span>
          <small>Product image unavailable</small>
        </div>
      );
    }

    return (
      <div className="product-row__placeholder">
        <span>{getInitials(fallbackLabel)}</span>
      </div>
    );
  }

  return (
    <img
      src={src}
      alt={alt}
      loading={loading}
      decoding="async"
      onError={() => {
        failedImageUrls.add(src);
        setHasFailed(true);
      }}
    />
  );
}

export function ProductCard({ item, label, imageBaseUrl, onSelect, selected = false }: ProductCardProps) {
  const name = getItemText(item, "prod_name", item.article_id);
  const category = getItemText(item, "product_group_name", "Fashion item");
  const color = getItemText(item, "colour_group_name", "Unknown");
  const description = getItemText(item, "detail_desc", "No detail description is available.");
  const imageUrl = getItemImageUrl(item, imageBaseUrl);

  return (
    <button
      type="button"
      className={`product-row${selected ? " is-selected" : ""}`}
      onClick={() => onSelect(item)}
    >
      <div className="product-row__media" aria-hidden="true">
        <ProductImage src={imageUrl} alt={name} fallbackLabel={name} variant="card" />
      </div>

      <div className="product-row__body">
        <div className="product-row__header">
          <div className="product-row__eyebrow">
            <span className="product-row__label">{label}</span>
            <span className="product-row__source">{item.source}</span>
          </div>
          <strong className="product-row__score">{item.score.toFixed(3)}</strong>
        </div>

        <h3>{name}</h3>

        <div className="product-row__meta">
          <span>{category}</span>
          <span>{color}</span>
          <span>{item.article_id}</span>
        </div>

        <p>{shrinkText(description, 140)}</p>
      </div>
    </button>
  );
}
