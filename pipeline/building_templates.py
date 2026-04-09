"""Localized templates for building condition text generation.

Templates use {name} and {items} placeholders resolved at generation time.
Game entity names (cultures, religions, advances, etc.) come from loc files.
These templates cover only the structural/glue text that is NOT in game loc files.
"""

from __future__ import annotations

# 11 language codes matching languages.py
_L = ("en", "ja", "de", "es", "fr", "ko", "pl", "pt-br", "ru", "tr", "zh-hans")


def _t(*vals: str) -> dict[str, str]:
    """Build {lang: text} dict from positional args matching _L order."""
    return dict(zip(_L, vals))


# ── Flag templates (no parameters) ──

TEMPLATES: dict[str, dict[str, str]] = {
    "flag.is_special": _t(
        "Special Building", "特殊建物",
        "Spezialgebäude", "Edificio especial", "Bâtiment spécial",
        "특수 건물", "Budowla specjalna", "Edifício especial",
        "Особое сооружение", "Özel yapı", "特殊建筑",
    ),
    "flag.is_foreign": _t(
        "Foreign Building", "外国建物",
        "Ausländisches Gebäude", "Edificio extranjero", "Bâtiment étranger",
        "외국 건물", "Obca budowla", "Edifício estrangeiro",
        "Иностранное сооружение", "Yabancı yapı", "外国建筑",
    ),
    "flag.no_estates": _t(
        "Cannot be built as Estate", "荘園建設不可",
        "Nicht als Anwesen baubar", "No construible como propiedad",
        "Non constructible en tant que domaine",
        "영지로 건설 불가", "Nie można budować jako posiadłość",
        "Não pode ser construído como propriedade",
        "Нельзя построить как поместье", "Mülk olarak inşa edilemez",
        "不能作为庄园建造",
    ),

    # ── Location boolean templates (no parameters) ──

    "loc_bool.is_capital.true": _t(
        "Capital only", "首都のみ",
        "Nur Hauptstadt", "Solo capital", "Capitale uniquement",
        "수도만", "Tylko stolica", "Apenas capital",
        "Только столица", "Yalnızca başkent", "仅首都",
    ),
    "loc_bool.is_capital.false": _t(
        "Non-capital only", "首都以外",
        "Nur Nicht-Hauptstadt", "Solo no capital", "Hors capitale uniquement",
        "수도 외", "Tylko poza stolicą", "Apenas fora da capital",
        "Только не столица", "Başkent dışı", "仅非首都",
    ),
    "loc_bool.is_port.true": _t(
        "Port city only", "港湾都市のみ",
        "Nur Hafenstadt", "Solo ciudad portuaria", "Port uniquement",
        "항구 도시만", "Tylko miasto portowe", "Apenas cidade portuária",
        "Только порт", "Yalnızca liman kenti", "仅港口城市",
    ),
    "loc_bool.is_coastal.true": _t(
        "Coastal only", "沿岸のみ",
        "Nur Küste", "Solo costero", "Côtier uniquement",
        "해안만", "Tylko wybrzeże", "Apenas litoral",
        "Только побережье", "Yalnızca kıyı", "仅沿海",
    ),
    "loc_bool.is_coastal.false": _t(
        "Inland only", "内陸のみ",
        "Nur Binnenland", "Solo interior", "Intérieur uniquement",
        "내륙만", "Tylko wnętrze lądu", "Apenas interior",
        "Только внутренние", "Yalnızca iç bölge", "仅内陆",
    ),
    "loc_bool.has_river.true": _t(
        "River only", "河川沿いのみ",
        "Nur am Fluss", "Solo junto a río", "Rivière uniquement",
        "하천만", "Tylko nad rzeką", "Apenas junto a rio",
        "Только у реки", "Yalnızca nehir kenarı", "仅河流沿岸",
    ),
    "loc_bool.is_adjacent_to_lake.true": _t(
        "Lakeside only", "湖沿いのみ",
        "Nur am See", "Solo junto a lago", "Lac uniquement",
        "호수만", "Tylko nad jeziorem", "Apenas junto a lago",
        "Только у озера", "Yalnızca göl kenarı", "仅湖泊沿岸",
    ),
    "loc_bool.is_market_center.true": _t(
        "Market center only", "市場中心地のみ",
        "Nur Marktzentrum", "Solo centro de mercado", "Centre de marché uniquement",
        "시장 중심지만", "Tylko centrum handlowe", "Apenas centro de mercado",
        "Только рыночный центр", "Yalnızca pazar merkezi", "仅市场中心",
    ),
    "loc_bool.has_road_to_capital.true": _t(
        "Road connection to capital required", "首都への道路接続が必要",
        "Straßenverbindung zur Hauptstadt nötig",
        "Conexión por carretera a la capital requerida",
        "Connexion routière à la capitale requise",
        "수도로의 도로 연결 필요",
        "Wymagane połączenie drogowe ze stolicą",
        "Conexão rodoviária à capital necessária",
        "Требуется дорога к столице",
        "Başkente yol bağlantısı gerekli",
        "需要通往首都的道路连接",
    ),
    "loc_bool.is_overseas_for_owner.true": _t(
        "Overseas only", "海外領のみ",
        "Nur Übersee", "Solo ultramar", "Outre-mer uniquement",
        "해외 영토만", "Tylko zamorskie", "Apenas ultramar",
        "Только заморские", "Yalnızca denizaşırı", "仅海外领地",
    ),
    "loc_bool.is_overseas_for_owner.false": _t(
        "Domestic only", "本土のみ",
        "Nur Inland", "Solo doméstico", "Métropole uniquement",
        "본토만", "Tylko krajowe", "Apenas doméstico",
        "Только метрополия", "Yalnızca anavatanda", "仅本土",
    ),

    # ── Event-only ──

    "event_only": _t(
        "Event only (cannot be built normally)", "イベント限定（通常建設不可）",
        "Nur durch Ereignis (nicht normal baubar)",
        "Solo por evento (no construible normalmente)",
        "Événement uniquement (non constructible normalement)",
        "이벤트 전용 (일반 건설 불가)",
        "Tylko wydarzenie (nie można normalnie zbudować)",
        "Apenas evento (não pode ser construído normalmente)",
        "Только событие (нельзя построить обычным способом)",
        "Yalnızca olay (normal inşa edilemez)",
        "仅限事件（无法正常建造）",
    ),

    # ── Reference templates (with {name} parameter) ──

    "ref.government_type": _t(
        "Government: {name}", "政体: {name}",
        "Regierungsform: {name}", "Gobierno: {name}", "Gouvernement : {name}",
        "정체: {name}", "Ustrój: {name}", "Governo: {name}",
        "Форма правления: {name}", "Yönetim: {name}", "政体：{name}",
    ),
    "ref.has_reform": _t(
        "Government Reform: {name}", "政体改革: {name}",
        "Regierungsreform: {name}", "Reforma de gobierno: {name}",
        "Réforme gouvernementale : {name}",
        "정부 개혁: {name}", "Reforma rządu: {name}", "Reforma governamental: {name}",
        "Реформа правления: {name}", "Yönetim reformu: {name}", "政府改革：{name}",
    ),
    "ref.has_advance": _t(
        "Advance: {name}", "進歩: {name}",
        "Fortschritt: {name}", "Avance: {name}", "Progrès : {name}",
        "진보: {name}", "Postęp: {name}", "Avanço: {name}",
        "Открытие: {name}", "İlerleme: {name}", "进步：{name}",
    ),
    "ref.culture": _t(
        "Culture: {name}", "文化: {name}",
        "Kultur: {name}", "Cultura: {name}", "Culture : {name}",
        "문화: {name}", "Kultura: {name}", "Cultura: {name}",
        "Культура: {name}", "Kültür: {name}", "文化：{name}",
    ),
    "ref.dominant_culture": _t(
        "Dominant culture: {name}", "支配文化: {name}",
        "Dominante Kultur: {name}", "Cultura dominante: {name}",
        "Culture dominante : {name}",
        "지배 문화: {name}", "Dominująca kultura: {name}",
        "Cultura dominante: {name}",
        "Доминирующая культура: {name}", "Baskın kültür: {name}",
        "主导文化：{name}",
    ),
    "ref.has_culture_group": _t(
        "Culture group: {name}", "文化グループ: {name}",
        "Kulturgruppe: {name}", "Grupo cultural: {name}",
        "Groupe culturel : {name}",
        "문화 집단: {name}", "Grupa kulturowa: {name}",
        "Grupo cultural: {name}",
        "Культурная группа: {name}", "Kültür grubu: {name}",
        "文化组：{name}",
    ),
    "ref.religion": _t(
        "Religion: {name}", "宗教: {name}",
        "Religion: {name}", "Religión: {name}", "Religion : {name}",
        "종교: {name}", "Religia: {name}", "Religião: {name}",
        "Религия: {name}", "Din: {name}", "宗教：{name}",
    ),
    "ref.religion_group": _t(
        "Religion group: {name}", "宗教グループ: {name}",
        "Religionsgruppe: {name}", "Grupo religioso: {name}",
        "Groupe religieux : {name}",
        "종교 집단: {name}", "Grupa religijna: {name}",
        "Grupo religioso: {name}",
        "Религиозная группа: {name}", "Din grubu: {name}",
        "宗教组：{name}",
    ),
    "ref.tag": _t(
        "Country: {name}", "国家: {name}",
        "Land: {name}", "País: {name}", "Pays : {name}",
        "국가: {name}", "Kraj: {name}", "País: {name}",
        "Страна: {name}", "Ülke: {name}", "国家：{name}",
    ),
    "ref.has_or_had_tag": _t(
        "Country: {name}", "国家: {name}",
        "Land: {name}", "País: {name}", "Pays : {name}",
        "국가: {name}", "Kraj: {name}", "País: {name}",
        "Страна: {name}", "Ülke: {name}", "国家：{name}",
    ),
    "ref.has_policy": _t(
        "Policy: {name}", "政策: {name}",
        "Politik: {name}", "Política: {name}", "Politique : {name}",
        "정책: {name}", "Polityka: {name}", "Política: {name}",
        "Политика: {name}", "Politika: {name}", "政策：{name}",
    ),
    "ref.location": _t(
        "Specific location: {name}", "特定地域: {name}",
        "Bestimmter Ort: {name}", "Ubicación específica: {name}",
        "Emplacement spécifique : {name}",
        "특정 지역: {name}", "Konkretna lokalizacja: {name}",
        "Localização específica: {name}",
        "Определённое место: {name}", "Belirli konum: {name}",
        "特定地点：{name}",
    ),
    "ref.owns_location": _t(
        "Owns location: {name}", "所有地域: {name}",
        "Besitzt Ort: {name}", "Posee ubicación: {name}",
        "Possède l'emplacement : {name}",
        "지역 소유: {name}", "Posiada lokalizację: {name}",
        "Possui localização: {name}",
        "Владеет местом: {name}", "Konuma sahip: {name}",
        "拥有地点：{name}",
    ),
    "ref.market_produces": _t(
        "Market produces: {name}", "市場で生産中: {name}",
        "Markt produziert: {name}", "Mercado produce: {name}",
        "Le marché produit : {name}",
        "시장 생산: {name}", "Rynek produkuje: {name}",
        "Mercado produz: {name}",
        "Рынок производит: {name}", "Pazar üretir: {name}",
        "市场生产：{name}",
    ),

    # ── Terrain templates (with {name} parameter) ──

    "terrain.vegetation": _t(
        "Vegetation: {name}", "植生: {name}",
        "Vegetation: {name}", "Vegetación: {name}", "Végétation : {name}",
        "식생: {name}", "Roślinność: {name}", "Vegetação: {name}",
        "Растительность: {name}", "Bitki örtüsü: {name}", "植被：{name}",
    ),
    "terrain.topography": _t(
        "Topography: {name}", "地形: {name}",
        "Topographie: {name}", "Topografía: {name}", "Topographie : {name}",
        "지형: {name}", "Topografia: {name}", "Topografia: {name}",
        "Топография: {name}", "Topoğrafya: {name}", "地形：{name}",
    ),
    "terrain.climate": _t(
        "Climate: {name}", "気候: {name}",
        "Klima: {name}", "Clima: {name}", "Climat : {name}",
        "기후: {name}", "Klimat: {name}", "Clima: {name}",
        "Климат: {name}", "İklim: {name}", "气候：{name}",
    ),

    # ── Logic compound templates ──

    "logic.or": _t(
        "Any of: {items}", "いずれか: {items}",
        "Eines von: {items}", "Cualquiera de: {items}",
        "L'un de : {items}",
        "다음 중 하나: {items}", "Dowolne z: {items}",
        "Qualquer um de: {items}",
        "Любое из: {items}", "Herhangi biri: {items}",
        "以下任一：{items}",
    ),
    "logic.not": _t(
        "Excluded: {items}", "除外: {items}",
        "Ausgeschlossen: {items}", "Excluido: {items}",
        "Exclu : {items}",
        "제외: {items}", "Wykluczone: {items}",
        "Excluído: {items}",
        "Исключено: {items}", "Hariç: {items}",
        "排除：{items}",
    ),

    # ── Modifier check ──

    "modifier_check": _t(
        "Requires modifier: {name}", "要補正: {name}",
        "Erfordert Modifikator: {name}", "Requiere modificador: {name}",
        "Nécessite un modificateur : {name}",
        "보정 필요: {name}", "Wymaga modyfikatora: {name}",
        "Requer modificador: {name}",
        "Требуется модификатор: {name}", "Değiştirici gerekli: {name}",
        "需要修正值：{name}",
    ),

    # ── Country boolean ──

    "country_bool.has_slavery": _t(
        "Requires Slavery", "奴隷制が必要",
        "Erfordert Sklaverei", "Requiere esclavitud",
        "Nécessite l'esclavage",
        "노예제 필요", "Wymaga niewolnictwa",
        "Requer escravidão",
        "Требуется рабство", "Kölelik gerekli",
        "需要奴隶制",
    ),

    # ── Trade range ──

    "ref.in_trade_range": _t(
        "Within trade range", "交易圏内であること",
        "In Handelsreichweite", "Dentro del rango comercial",
        "Dans la portée commerciale",
        "교역 범위 내", "W zasięgu handlu",
        "Dentro do alcance comercial",
        "В торговой зоне", "Ticaret menzilinde",
        "在贸易范围内",
    ),
}

# ── Terrain value translations (not in game loc files) ──

TERRAIN_VALUES: dict[str, dict[str, str]] = {
    # Vegetation
    "woods": _t(
        "Woodlands", "森林",
        "Wälder", "Bosques", "Bois",
        "삼림", "Lasy", "Bosques",
        "Леса", "Ormanlar", "林地",
    ),
    "forest": _t(
        "Dense Forest", "深林",
        "Dichter Wald", "Bosque denso", "Forêt dense",
        "밀림", "Gęsty las", "Floresta densa",
        "Густой лес", "Sık orman", "密林",
    ),
    "jungle": _t(
        "Jungle", "密林",
        "Dschungel", "Selva", "Jungle",
        "정글", "Dżungla", "Selva",
        "Джунгли", "Cengel", "丛林",
    ),
    "sparse": _t(
        "Sparse", "疎林",
        "Spärlich", "Escaso", "Clairsemé",
        "성긴 숲", "Rzadka", "Esparsa",
        "Редколесье", "Seyrek", "稀疏植被",
    ),
    "farmland": _t(
        "Farmland", "農地",
        "Ackerland", "Tierras de cultivo", "Terres agricoles",
        "농지", "Ziemia uprawna", "Terras agrícolas",
        "Пашня", "Tarım arazisi", "农田",
    ),
    "grasslands": _t(
        "Grasslands", "草原",
        "Grasland", "Praderas", "Prairies",
        "초원", "Łąki", "Pradarias",
        "Степь", "Otlaklar", "草原",
    ),
    "desert": _t(
        "Desert", "砂漠",
        "Wüste", "Desierto", "Désert",
        "사막", "Pustynia", "Deserto",
        "Пустыня", "Çöl", "沙漠",
    ),
    # Topography
    "mountains": _t(
        "Mountains", "山岳",
        "Berge", "Montañas", "Montagnes",
        "산악", "Góry", "Montanhas",
        "Горы", "Dağlar", "山脉",
    ),
    "plateau": _t(
        "Plateau", "高原",
        "Hochebene", "Meseta", "Plateau",
        "고원", "Wyżyna", "Planalto",
        "Плато", "Yayla", "高原",
    ),
    "hills": _t(
        "Hills", "丘陵",
        "Hügel", "Colinas", "Collines",
        "구릉", "Wzgórza", "Colinas",
        "Холмы", "Tepeler", "丘陵",
    ),
    "wetlands": _t(
        "Wetlands", "湿地",
        "Feuchtgebiet", "Humedal", "Marécage",
        "습지", "Mokradła", "Pântano",
        "Болота", "Sulak arazi", "湿地",
    ),
    # Climate
    "mediterranean": _t(
        "Mediterranean", "地中海性",
        "Mediterran", "Mediterráneo", "Méditerranéen",
        "지중해성", "Śródziemnomorski", "Mediterrâneo",
        "Средиземноморский", "Akdeniz", "地中海",
    ),
    "continental": _t(
        "Continental", "大陸性",
        "Kontinental", "Continental", "Continental",
        "대륙성", "Kontynentalny", "Continental",
        "Континентальный", "Karasal", "大陆性",
    ),
    "oceanic": _t(
        "Oceanic", "海洋性",
        "Ozeanisch", "Oceánico", "Océanique",
        "해양성", "Oceaniczny", "Oceânico",
        "Океанический", "Okyanusal", "海洋性",
    ),
    "tropical": _t(
        "Tropical", "熱帯",
        "Tropisch", "Tropical", "Tropical",
        "열대", "Tropikalny", "Tropical",
        "Тропический", "Tropikal", "热带",
    ),
    "arid": _t(
        "Arid", "乾燥",
        "Trocken", "Árido", "Aride",
        "건조", "Suchy", "Árido",
        "Засушливый", "Kurak", "干旱",
    ),
}


# ── Item separator for compound templates ──

ITEM_SEP: dict[str, str] = _t(
    " / ", " / ",
    " / ", " / ", " / ",
    " / ", " / ", " / ",
    " / ", " / ", " / ",
)

NOT_SEP: dict[str, str] = _t(
    ", ", "、",
    ", ", ", ", ", ",
    ", ", ", ", ", ",
    ", ", ", ", "、",
)
