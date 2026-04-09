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

/** Structured requirement node for building conditions. */
export interface RequirementNode {
  type: string;
  scope: string;
  flag?: string;
  check?: string;
  value?: boolean;
  kind?: string;
  refs?: string[];
  terrain_type?: string;
  values?: string[];
  modifier?: string;
  children?: RequirementNode[];
}

export interface ModifierEntry {
  key: string;
  value: number | boolean;
}

export interface ProductionMethodCore {
  category?: string;
  goods?: Record<string, number>;
  produced?: string;
  output?: number;
}

export interface BuildingItem {
  id: string;
  category?: string;
  pop_type?: string;
  expensive?: boolean;
  max_levels?: number;
  max_levels_scaling?: string[];
  max_levels_raw?: string;
  build_days?: number;
  build_time?: string;
  settlements?: string[];
  requirements?: RequirementNode[];
  facets?: string[];
  modifier?: ModifierEntry[];
  raw_modifier?: ModifierEntry[];
  production_methods?: Record<string, ProductionMethodCore>;
  construction_demand?: string;
  icon: string;
  source_file: string;
}

export interface ReligionModifier {
  key: string;
  value: number | string;
  bool?: boolean;
  scaled?: boolean;
  pct?: boolean;
  inv?: boolean;
}

export interface ReligionItem {
  id: string;
  group_id: string;
  modifier?: ReligionModifier[];
  opinions?: Record<string, string>;
  religious_aspects?: number;
  max_sects?: number;
  enable?: string;
  language?: string;
  mechanics?: string[];
  icon: string;
  source_file: string;
}

export interface HolySiteModifier {
  key: string;
  value: number | string;
  pct?: boolean;
  inv?: boolean;
}

export interface HolySiteType {
  id: string;
  location_modifier?: HolySiteModifier[];
  country_modifier?: HolySiteModifier[];
}

export interface HolySiteItem {
  id: string;
  location: string;
  type: string;
  importance: number;
  religions: string[];
  source_file: string;
}

export interface AspectModifier {
  key: string;
  value: number | string;
  bool?: boolean;
  scaled?: boolean;
  pct?: boolean;
  inv?: boolean;
}

export interface AspectItem {
  id: string;
  religions: string[];
  icon: string;
  modifier?: AspectModifier[];
  opinions?: Record<string, number>;
  excludes?: string[];
  source_file: string;
}

export interface CountryItem {
  tag: string;
  file_region: string;
  culture_definition?: string;
  religion_definition?: string;
  description_category?: string;
  difficulty?: number;
  is_historic?: boolean;
  culture_groups?: string[];
  capital?: string | string[];
  country_rank?: string;
  starting_tech?: number;
  government_type?: string;
  heir_selection?: string;
  color?: number[];
  is_formable: boolean;
  formable_level?: number;
  formable_rule?: string;
  icon: string;
  source_file: string;
}

/** Per-building localization entry (extended beyond standard name/desc). */
export interface BuildingLocEntry {
  name: string;
  desc?: string;
  condition_lines?: string[];
  modifiers?: Record<string, { name: string; desc?: string }>;
  raw_modifiers?: Record<string, { name: string; desc?: string }>;
  pm?: Record<string, { name: string; category?: string }>;
}
