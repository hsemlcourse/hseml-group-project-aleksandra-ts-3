NAVY = '#1B262C'
TEAL = '#517D8B'
SKY = '#C9D9F3'
LAVENDER = '#E8EBF7'
WHITE = '#FFFFFF'

RISK_COLORS = {
    'низкий': TEAL,
    'средний': '#3A6B7C',
    'высокий': NAVY,
}
RISK_BG = {
    'низкий': LAVENDER,
    'средний': SKY,
    'высокий': '#D4E0EB',
}

THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Manrope', 'Segoe UI', sans-serif;
    color: {NAVY};
}}

.stApp {{
    background: linear-gradient(180deg, {WHITE} 0%, {LAVENDER} 100%);
}}

.block-container {{
    padding-top: 1.2rem;
    max-width: 1200px;
}}

.hero {{
    background: linear-gradient(135deg, {NAVY} 0%, {TEAL} 100%);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    color: {WHITE};
    margin-bottom: 1rem;
    box-shadow: 0 10px 28px rgba(27, 38, 44, 0.15);
}}

.hero h1, .hero h1 * {{
    margin: 0;
    font-size: 1.65rem;
    font-weight: 700;
    color: {WHITE} !important;
}}

.hero p, .hero p * {{
    margin: 0.4rem 0 0;
    font-size: 0.95rem;
    color: {WHITE} !important;
}}

.result-card {{
    background: {WHITE};
    border: 1px solid {SKY};
    border-radius: 18px;
    padding: 1.5rem 1.4rem 1.35rem;
    box-shadow: 0 12px 32px rgba(81, 125, 139, 0.14);
    text-align: center;
}}

.result-card.risk-низкий {{ border-top: 4px solid {TEAL}; }}
.result-card.risk-средний {{ border-top: 4px solid #3A6B7C; }}
.result-card.risk-высокий {{ border-top: 4px solid {NAVY}; }}

.result-card h3 {{
    color: {NAVY};
    margin: 0 0 1rem 0;
    font-size: 1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

.gauge {{
    --p: 20;
    --c: {TEAL};
    width: 152px;
    height: 152px;
    border-radius: 50%;
    margin: 0 auto 1rem;
    background: conic-gradient(var(--c) calc(var(--p) * 1%), {LAVENDER} 0);
    display: flex;
    align-items: center;
    justify-content: center;
}}

.gauge-inner {{
    width: 118px;
    height: 118px;
    border-radius: 50%;
    background: {WHITE};
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 16px rgba(27, 38, 44, 0.08);
}}

.gauge-value {{
    font-size: 1.85rem;
    font-weight: 700;
    color: var(--c);
    line-height: 1.1;
}}

.gauge-label {{
    font-size: 0.72rem;
    color: {TEAL};
    margin-top: 0.15rem;
}}

.risk-pill {{
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.88rem;
}}

.result-meta {{
    background: {LAVENDER};
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
    margin-top: 0.85rem;
    text-align: left;
    font-size: 0.9rem;
    line-height: 1.45;
    color: {NAVY};
}}

.result-meta b {{
    color: {TEAL};
}}

.panel-soft {{
    background: {LAVENDER};
    border: 1px solid {SKY};
    border-radius: 10px;
    padding: 0.85rem 1rem;
    color: {NAVY};
    font-size: 0.9rem;
}}

div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {{
    background: {TEAL} !important;
    color: {WHITE} !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {{
    background: {NAVY} !important;
    color: {WHITE} !important;
}}
</style>
"""
