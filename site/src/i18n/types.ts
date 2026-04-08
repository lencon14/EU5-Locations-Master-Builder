/** Shared type definitions for core data items. */

export interface GoodsItem {
  id: string;
  category: string;
  method?: string;
  default_market_price: number;
  transport_cost?: number;
  base_production?: number;
  origin?: string;
  food?: number;
  development_threshold?: number;
  tags?: string[];
  demand_add?: Record<string, number>;
  demand_multiply?: Record<string, number>;
  wealth_impact_threshold?: Record<string, number>;
  icon: string;
  source_file: string;
}
