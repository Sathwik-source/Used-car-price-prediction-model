import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── LOAD MODELS ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("model_with_columns.pkl", "rb") as f:
        pkl_data = pickle.load(f)
    xgb_pipeline = joblib.load("xgb_pipeline.joblib")
    return pkl_data["model"], pkl_data["columns"], xgb_pipeline

model, columns, xgb_pipeline = load_models()

brands       = sorted([c.replace("brand_", "")        for c in columns if c.startswith("brand_")])
fuel_types   = sorted([c.replace("fuel_", "")         for c in columns if c.startswith("fuel_")])
trans_types  = sorted([c.replace("transmission_", "") for c in columns if c.startswith("transmission_")])
seller_types = sorted([c.replace("seller_type_", "")  for c in columns if c.startswith("seller_type_")])
owner_types  = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]
owner_types  = ["First Owner"] + [o for o in ["Second Owner", "Third Owner", "Fourth & Above Owner"] if f"owner_{o}" in columns]

# ═══════════════════════════════════════════════════════════
# BRAND LOGOS — genuine emblem geometry, inline SVG
# Every logo uses the brand's real primary color and emblem
# shape. No text-on-rectangle approach.
# ═══════════════════════════════════════════════════════════
BRAND_SVG = {
    "Maruti": """<svg viewBox="0 0 60 40" xmlns="http://www.w3.org/2000/svg">
  <rect width="60" height="40" fill="#E60012"/>
  <g transform="translate(16.000,6.000) scale(1.16667)" fill="#FFFFFF">
    <path d="M17.369 19.995C13.51 22.39 12 24 12 24L.105 15.705s5.003-3.715 9.186-.87l5.61 3.882.683-.453L.106 7.321s2.226-.65 6.524-3.315C10.49 1.609 12 0 12 0l11.895 8.296s-5.003 3.715-9.187.87L9.1 5.281l-.683.454L23.893 16.68s-2.224.649-6.524 3.315Z"/>
  </g>
</svg>""",

    "Hyundai": """<svg viewBox="0 0 60 40" xmlns="http://www.w3.org/2000/svg">
  <rect width="60" height="40" fill="#002C5F"/>
  <g transform="translate(16.000,6.000) scale(1.16667)" fill="#FFFFFF">
    <path d="M12 18.1622c-6.6275 0-12-2.7586-12-6.163 0-3.4028 5.3725-6.1614 12-6.1614 6.6278 0 12 2.7586 12 6.1614 0 3.4044-5.3722 6.163-12 6.163zM7.6023 7.17C3.701 7.9784.973 9.8302.973 11.9844c0 1.1929.8382 2.2932 2.248 3.1757.1174.0724.1941.0862.251.0826.1019-.006.1593-.0698.201-.146.028-.0485.0631-.1225.0972-.1968.4601-1.0834 2.0776-4.8333 4.2023-7.3758a1.1775 1.1775 0 0 0 .1048-.1461c.046-.084.0356-.1513.0006-.192-.0593-.0647-.2247-.065-.4756-.016zM9.742 8.8995c-1.1728 2.8492 1.0473 2.4961 1.6478 2.3637 1.0203-.2258 1.9944-.6128 2.7746-.925 2.2216-.8887 3.4012-1.7804 3.7925-2.123a1.9839 1.9839 0 0 0 .1076-.0988c.0557-.058.0976-.1192.0976-.2002 0-.0936-.081-.1687-.2374-.2231-.012-.0049-.0517-.021-.0641-.025-1.698-.5415-3.724-.8563-5.9016-.8563-.0168 0-.0586-.0022-.1169 0-.2608.0078-.5509.0664-.787.1888-.7777.4049-1.1163 1.4235-1.313 1.899zm10.5851.0037c-.0268.0487-.0612.1224-.0962.1974-.4599 1.0826-2.0774 4.831-4.2018 7.3733-.0515.063-.0796.1031-.1042.1467-.0492.0846-.0388.1535 0 .1935.0572.0641.2235.0654.474.0157 3.8998-.81 6.628-2.6606 6.628-4.8149 0-1.1925-.836-2.2928-2.2472-3.1745-.1161-.073-.1934-.0871-.25-.083-.1028.0067-.16.0699-.2026.1458zM14.258 15.099c1.173-2.849-1.0483-2.494-1.6467-2.3622-1.0218.225-1.996.613-2.7757.924-2.2226.8883-3.4017 1.782-3.7944 2.1234-.0468.0428-.0833.0742-.1066.0995-.0564.0573-.0967.1178-.0967.2007 0 .0923.08.1688.2362.2229.012.0048.0511.0213.0657.0255 1.696.54 3.722.8557 5.9.8557.0177 0 .0592.0016.1178 0 .2609-.0081.5522-.0677.7871-.1888.7781-.4052 1.1169-1.4234 1.3133-1.9007z"/>
  </g>
</svg>""",

    "Tata": """<svg viewBox="0 0 60 40" xmlns="http://www.w3.org/2000/svg">
  <rect width="60" height="40" fill="#00205B"/>
  <ellipse cx="30" cy="13" rx="20" ry="9"
           fill="none" stroke="#FFFFFF" stroke-width="3.2"/>
  <rect x="27.2" y="22" width="5.6" height="15" rx="1" fill="#FFFFFF"/>
</svg>""",

    "Honda": """<svg viewBox="0 0 60 40" xmlns="http://www.w3.org/2000/svg">
  <rect width="60" height="40" fill="#CC0000"/>
  <g transform="translate(16.000,6.000) scale(1.16667)" fill="#FFFFFF">
    <path d="M23.902 6.87c-.33-3.218-2.47-3.895-4.354-4.204-.946-.16-2.63-.3-3.716-.34-.946-.06-3.168-.09-3.835-.09-.657 0-2.89.03-3.835.09-1.076.04-2.77.18-3.716.34C2.563 2.985.42 3.66.092 6.87c-.08.877-.1 2.023-.09 3.248.03 2.031.2 3.406.3 4.363.07.657.338 2.62.687 3.636.478 1.395.916 1.803 1.424 2.222.937.757 2.471.996 2.79 1.056 1.733.31 5.24.368 6.784.368 1.544 0 5.05-.05 6.784-.368.329-.06 1.863-.29 2.79-1.056.508-.419.946-.827 1.424-2.222.35-1.016.628-2.979.698-3.636.1-.957.279-2.332.299-4.363.04-1.225.01-2.371-.08-3.248m-1.176 5.4c-.19 2.57-.418 4.104-.747 5.22-.29.976-.637 1.623-1.165 2.092-.867.787-2.063.956-2.76 1.056-1.514.23-4.055.3-6.057.3-2.002 0-4.543-.08-6.057-.3-.697-.1-1.893-.269-2.76-1.056-.518-.469-.876-1.126-1.155-2.093-.329-1.105-.558-2.65-.747-5.22-.11-1.543-.09-4.054.08-5.4.258-2.011 1.255-3.018 3.387-3.396.996-.18 2.34-.31 3.606-.37 1.016-.07 2.7-.1 3.636-.09.936-.01 2.62.03 3.636.09 1.275.06 2.61.19 3.606.37 2.142.378 3.139 1.395 3.388 3.397.199 1.345.229 3.856.11 5.4m-5.202-8.39c-.548 2.462-.767 3.588-1.216 5.37-.428 1.715-.767 3.298-1.335 4.065-.587.777-1.365.947-1.893 1.006-.279.03-.478.04-1.066.05-.596 0-.796-.02-1.075-.05-.528-.06-1.315-.229-1.892-1.006-.578-.767-.907-2.35-1.335-4.064-.47-1.773-.678-2.91-1.236-5.37 0 0-.548.02-.797.04-.329.02-.588.05-.867.09.343 5.372.692 11.079 1.126 16.13a21.983 21.983 0 002.39.169c.33-1.266.748-3.02 1.207-3.767.378-.608.966-.677 1.295-.717.518-.07.956-.08 1.165-.08.2-.01.637 0 1.165.08.33.05.917.11 1.295.717.47.747.877 2.5 1.206 3.766 0 0 .358-.01 1.165-.05.41-.018.82-.058 1.226-.12.458-5.39.785-10.728 1.126-16.128-.28-.04-.538-.07-.867-.09-.23-.02-.787-.04-.787-.04z"/>
  </g>
</svg>""",

    "Toyota": """<svg viewBox="0 0 60 40" xmlns="http://www.w3.org/2000/svg">
  <rect width="60" height="40" fill="#EB0A1E"/>
  <g transform="translate(16.000,6.000) scale(1.16667)" fill="#FFFFFF">
    <path d="M12 3.848C5.223 3.848 0 7.298 0 12c0 4.702 5.224 8.152 12 8.152S24 16.702 24 12c0-4.702-5.223-8.152-12-8.152zm7.334 3.839c0 1.08-1.725 1.913-4.488 2.246-.26-2.58-1.005-4.279-1.963-4.913 2.948.184 6.45 1.227 6.45 2.667zM12 16.401c-.96 0-1.746-1.5-1.808-4.389.577.047 1.18.072 1.808.072.628 0 1.23-.025 1.807-.072-.061 2.89-.847 4.389-1.807 4.389zm0-6.308c-.59 0-1.155-.019-1.69-.054.261-1.728.92-3.15 1.69-3.15.77 0 1.428 1.422 1.689 3.15-.535.034-1.099.054-1.689.054zm-.882-5.075c-.956.633-1.706 2.333-1.964 4.915C6.391 9.6 4.665 8.767 4.665 7.687c0-1.44 3.504-2.49 6.453-2.669zM2.037 11.68a5.265 5.265 0 011.048-3.164c.27 1.547 2.522 2.881 5.972 3.37V12c0 3.772.879 6.203 2.087 6.97-5.107-.321-9.107-3.48-9.107-7.29zm10.823 7.29c1.207-.767 2.087-3.198 2.087-6.97v-.115c3.447-.488 5.704-1.826 5.972-3.37a5.26 5.26 0 011.049 3.165c-.004 3.81-4.008 6.969-9.109 7.29z"/>
  </g>
</svg>""",

    "Ford": """<svg viewBox="0 0 60 40" xmlns="http://www.w3.org/2000/svg">
  <rect width="60" height="40" fill="#003399"/>
  <g transform="translate(16.000,6.000) scale(1.16667)" fill="#FFFFFF">
    <path d="M12 8.236C5.872 8.236.905 9.93.905 12.002S5.872 15.767 12 15.767c6.127 0 11.094-1.693 11.094-3.765 0-2.073-4.967-3.766-11.094-3.766zm-5.698 6.24c-.656.005-1.233-.4-1.3-1.101a1.415 1.415 0 0 1 .294-1.02c.195-.254.525-.465.804-.517.09-.017.213-.006.264.054.079.093.056.194-.023.234-.213.109-.47.295-.597.55a.675.675 0 0 0 .034.696c.263.397.997.408 1.679-.225.169-.156.32-.304.473-.48.3-.344.4-.47.8-1.024.005-.006.006-.014.004-.018-.003-.007-.009-.01-.02-.01-.267.007-.5.087-.725.255-.065.048-.159.041-.2-.021-.046-.07-.013-.163.062-.215.363-.253.76-.298 1.166-.367 0 0 .028.002.051-.03.167-.213.292-.405.47-.621.178-.22.41-.42.586-.572.246-.212.404-.283.564-.37.043-.022-.005-.049-.018-.049-.896-.168-1.827-.386-2.717-.056-.616.23-.887.718-.757 1.045.093.231.397.27.683.13a1.55 1.55 0 0 0 .611-.544c.087-.134.27-.038.171.195-.26.611-.757 1.097-1.363 1.118-.516.016-.849-.363-.848-.831.002-.924 1.03-1.532 2.11-1.622 1.301-.108 2.533.239 3.825.395.989.12 1.938.123 2.932-.106.118-.025.2.05.193.168-.01.172-.143.337-.47.516-.373.204-.763.266-1.17.27-.984.008-1.901-.376-2.85-.582.002.041.012.091-.023.117-.525.388-1 .782-1.318 1.334-.011.013-.005.025.013.024.277-.015.525-.022.783-.042.045-.004.047-.015.043-.048a.64.64 0 0 1 .2-.558c.172-.153.387-.17.53-.06.16.126.147.353.058.523a.63.63 0 0 1-.382.31s-.03.006-.026.034c.006.043.2.151.217.18.017.027.008.07-.021.102a.123.123 0 0 1-.095.045c-.033 0-.053-.012-.096-.035a.92.92 0 0 1-.27-.217c-.024-.031-.037-.032-.099-.029-.279.017-.714.059-1.009.096-.071.008-.082.022-.096.047-.47.775-.972 1.61-1.523 2.17-.592.6-1.083.758-1.604.762zM19.05 10.71c-.091.158-1.849 2.834-1.96 3.11-.035.088-.04.155-.004.204.092.124.297.051.425-.038.381-.262.645-.58.937-.858.017-.013.046-.018.065 0 .043.04.106.091.15.137a.04.04 0 0 1 .002.057 5.873 5.873 0 0 1-.904.911c-.47.364-.939.457-1.172.224a.508.508 0 0 1-.14-.316c-.002-.057-.031-.06-.058-.034-.278.275-.76.579-1.198.362-.366-.18-.451-.618-.383-.986.001-.008-.006-.06-.051-.03a1.28 1.28 0 0 1-.3.162.853.853 0 0 1-.366.077.518.518 0 0 1-.451-.253.759.759 0 0 1-.095-.347c-.001-.011-.017-.032-.033-.005-.3.457-.579.899-.875 1.363-.016.022-.03.036-.06.037l-.587.001c-.036 0-.053-.028-.034-.063.104-.2.674-1.03 1.06-1.736.107-.194.085-.294.019-.337-.083-.054-.248.027-.387.133-.379.287-.697.735-.859.935-.095.117-.185.291-.433.56-.391.425-.91.669-1.408.5a.848.848 0 0 1-.546-.58c-.015-.052-.044-.066-.073-.032-.08.1-.245.249-.383.342-.015.011-.052.033-.084.017a.851.851 0 0 1-.152-.199.07.07 0 0 1 .016-.08c.197-.173.305-.271.391-.38.064-.08.113-.17.17-.315.12-.302.393-.866.938-1.158a1.81 1.81 0 0 1 .652-.219c.1-.01.183.002.213.08.011.033.039.105.056.158.011.032.003.057-.035.071-.32.122-.643.311-.865.61-.253.338-.321.746-.152.98.123.17.322.2.514.139.29-.092.538-.363.666-.663.138-.329.16-.717.058-1.059-.016-.059-.001-.104.037-.136.077-.063.184-.112.215-.128a.14.14 0 0 1 .182.045c.106.157.163.378.17.607.006.049.026.05.05.025.19-.202.366-.418.568-.58.185-.147.422-.267.643-.262.286.006.428.2.419.546-.001.044.03.04.051.011a1.19 1.19 0 0 1 .24-.264c.198-.163.4-.236.611-.222.26.02.468.257.425.527a.53.53 0 0 1-.281.406.362.362 0 0 1-.405-.044.336.336 0 0 1-.096-.322c.005-.025-.027-.048-.054-.02-.254.264-.273.606-.107.76.183.17.458.056.658-.075.366-.239.65-.563.979-.813.218-.166.467-.314.746-.351a.87.87 0 0 1 .454.052c.2.081.326.25.342.396.004.043.036.048.063.01.158-.246 1.005-1.517 1.075-1.65.02-.041.044-.047.089-.047h.606c.035 0 .051.02.036.047zm-2.32 2.204a.053.053 0 0 0-.003.04c.003.02.03.04.056.05.01.003.015.01.004.032-.075.16-.143.252-.237.391a1.472 1.472 0 0 1-.3.325c-.178.147-.424.307-.628.2-.09-.047-.13-.174-.127-.276.004-.288.132-.584.369-.875.288-.355.607-.539.816-.438.216.103.148.354.05.55zm-5.949-1.881a.398.398 0 0 1 .132-.345c.057-.05.133-.062.18-.022.052.045.027.157-.026.234a.43.43 0 0 1-.245.177c-.018.004-.034-.004-.041-.044zM12 7.5C5.34 7.5 0 9.497 0 12c0 2.488 5.383 4.5 12 4.5s12-2.02 12-4.5-5.383-4.5-12-4.5zm0 8.608C5.649 16.108.5 14.27.5 12.002.5 9.733 5.65 7.895 12 7.895s11.498 1.838 11.498 4.107c0 2.268-5.148 4.106-11.498 4.106z"/>
  </g>
</svg>""",

    "Volkswagen": """<svg viewBox="0 0 60 40" xmlns="http://www.w3.org/2000/svg">
  <rect width="60" height="40" fill="#001E50"/>
  <g transform="translate(16.000,6.000) scale(1.16667)" fill="#FFFFFF">
    <path d="M12 0C5.36 0 0 5.36 0 12S5.36 24 12 24 24 18.64 24 12 18.64 0 12 0M12 1.41C13.2 1.41 14.36 1.63 15.43 2L12.13 9.13C12.09 9.17 12.09 9.26 12 9.26S11.91 9.17 11.87 9.13L8.57 2C9.64 1.63 10.8 1.42 12 1.42M6.9 2.74L10.72 10.97C10.8 11.14 10.89 11.19 11 11.19H13C13.12 11.19 13.2 11.14 13.29 10.97L17.06 2.74C18.64 3.64 20 4.93 20.96 6.47L15.6 16.84C15.56 16.93 15.5 16.97 15.47 16.97C15.39 16.97 15.39 16.89 15.34 16.84L13.29 12.3C13.2 12.13 13.12 12.09 13 12.09H11C10.89 12.09 10.8 12.13 10.71 12.3L8.66 16.84C8.61 16.89 8.62 16.97 8.53 16.97C8.44 16.97 8.44 16.89 8.4 16.84L3 6.47C3.94 4.93 5.32 3.64 6.9 2.74M2.06 8.53L8.23 20.53C8.31 20.7 8.4 20.83 8.62 20.83C8.83 20.83 8.91 20.7 9 20.53L11.87 14.14C11.91 14.06 11.96 14 12 14C12.09 14 12.09 14.1 12.13 14.14L15.04 20.53C15.13 20.7 15.21 20.83 15.43 20.83C15.64 20.83 15.73 20.7 15.81 20.53L22 8.53C22.37 9.6 22.59 10.76 22.59 12C22.54 17.79 17.79 22.59 12 22.59C6.21 22.59 1.46 17.79 1.46 12C1.46 10.8 1.67 9.65 2.06 8.53Z"/>
  </g>
</svg>""",

    "Kia": """<svg viewBox="0 0 60 40" xmlns="http://www.w3.org/2000/svg">
  <rect width="60" height="40" fill="#05141F"/>
  <g transform="translate(16.000,6.000) scale(1.16667)" fill="#FFFFFF">
    <path d="M13.923 14.175c0 .046.015.072.041.072a.123.123 0 0 0 .058-.024l7.48-4.854a.72.72 0 0 1 .432-.13h1.644c.252 0 .422.168.422.42v3.139c0 .38-.084.6-.42.801l-1.994 1.2a.137.137 0 0 1-.067.024c-.024 0-.048-.019-.048-.088v-3.663c0-.043-.012-.071-.041-.071a.113.113 0 0 0-.058.024l-5.466 3.551a.733.733 0 0 1-.42.127h-3.624c-.254 0-.422-.168-.422-.422V9.757c0-.033-.015-.064-.044-.064a.118.118 0 0 0-.057.024L7.732 11.88c-.036.024-.046.041-.046.058 0 .014.008.029.032.055l2.577 2.575c.034.034.058.06.058.089 0 .024-.039.043-.084.043H7.94c-.183 0-.324-.026-.423-.125l-1.562-1.56a.067.067 0 0 0-.048-.024.103.103 0 0 0-.048.015l-2.61 1.57a.72.72 0 0 1-.423.122H.425C.168 14.7 0 14.53 0 14.279v-3.08c0-.38.084-.6.422-.8L2.43 9.192a.103.103 0 0 1 .052-.016c.032 0 .048.03.048.1V13.4c0 .043.01.063.041.063a.144.144 0 0 0 .06-.024L9.407 9.36a.733.733 0 0 1 .446-.124h3.648c.252 0 .422.168.422.42l-.002 4.518z"/>
  </g>
</svg>""",

    "Mahindra": """<svg viewBox="0 0 60 40" xmlns="http://www.w3.org/2000/svg">
  <rect width="60" height="40" fill="#E31837"/>
  <g transform="translate(16.000,6.000) scale(1.16667)" fill="#FFFFFF">
    <path d="M5.145 11.311H6.78a.67.67 0 0 1 .674.66v1.509H5.009a.408.408 0 0 1-.41-.404v-.524a.38.38 0 0 1 .383-.375h1.354l-.144.306h-.998c-.043 0-.092.034-.092.081v.412c0 .047.049.082.092.082h1.73v-.99c0-.191-.169-.338-.357-.338H4.945l.2-.419zm13.427-.787v2.959h-2.383a.408.408 0 0 1-.41-.403v-1.11a.67.67 0 0 1 .675-.659h1.357l-.2.422h-.948c-.188 0-.357.147-.357.337v.91c0 .046.049.08.092.08h1.644v-2.536h.53zM10.2 13.483h.527v-1.51a.67.67 0 0 0-.674-.659H8.932l-.2.422h1.111c.188 0 .357.147.357.337v1.41zm-2.195-2.96v2.96h.527v-2.96h-.527zm-4.4 2.96h.527v-1.51a.67.67 0 0 0-.674-.659H0v2.169h.526v-1.669c0-.047.05-.081.093-.081h1.09c.043 0 .092.034.092.081v1.669h.527v-1.669c0-.047.049-.081.092-.081h.828c.188 0 .357.147.357.337v1.413zm17.72-2.172H20a.67.67 0 0 0-.674.66v1.509h.527v-1.41c0-.19.169-.337.357-.337h.914l.2-.422zm-6.753 0a.67.67 0 0 1 .675.66v1.509h-.527v-1.41c0-.19-.17-.337-.357-.337h-1.268v1.75h-.527v-2.169c.665 0 1.333-.003 2.004-.003zm-3.19.137.527-.306v2.338h-.526v-2.032zm.53-.609v-.322h-.526v.625l.526-.303zm9.782.472h1.632a.67.67 0 0 1 .674.66v1.509h-2.445a.408.408 0 0 1-.41-.404v-.524a.38.38 0 0 1 .383-.375h1.354l-.144.306h-.998c-.043 0-.092.034-.092.081v.412c0 .047.049.082.092.082h1.73v-.99c0-.191-.169-.338-.357-.338h-1.622l.203-.419z"/>
  </g>
</svg>""",

    "Renault": """<svg viewBox="0 0 60 40" xmlns="http://www.w3.org/2000/svg">
  <rect width="60" height="40" fill="#EFDF00"/>
  <g transform="translate(16.000,6.000) scale(1.16667)" fill="#141414">
    <path d="M17.463 11.99l-4.097-7.692-.924 1.707 3.213 5.985-5.483 10.283L4.69 11.99 11.096 0H9.27L2.882 11.99 9.269 24h1.807zm3.655 0L14.711 0h-1.807L6.517 11.99l4.117 7.712.904-1.707-3.193-6.005 5.463-10.263L19.29 11.99 12.904 24h1.807Z"/>
  </g>
</svg>""",

    "Nissan": """<svg viewBox="0 0 60 40" xmlns="http://www.w3.org/2000/svg">
  <rect width="60" height="40" fill="#FFFFFF"/>
  <g transform="translate(16.000,6.000) scale(1.16667)" fill="#808080">
    <path d="M20.576 14.955l-.01.028c-1.247 3.643-4.685 6.086-8.561 6.086-3.876 0-7.32-2.448-8.562-6.09l-.01-.029H.71v.329l1.133.133c.7.08.847.39 1.038.78l.048.096c1.638 3.495 5.204 5.752 9.08 5.752 3.877 0 7.443-2.257 9.081-5.747l.048-.095c.19-.39.338-.7 1.038-.781l1.134-.134v-.328zM3.443 9.012c1.247-3.643 4.686-6.09 8.562-6.09 3.876 0 7.319 2.447 8.562 6.09l.01.028h2.728v-.328l-1.134-.133c-.7-.081-.847-.39-1.038-.781l-.047-.096C19.448 4.217 15.88 1.96 12.005 1.96c-3.881 0-7.443 2.257-9.081 5.752l-.048.095c-.19.39-.338.7-1.038.781l-1.133.133v.329h2.724zm13.862 1.586l-1.743 2.795h.752l.31-.5h2.033l.31.5h.747l-1.743-2.795zm1.033 1.766h-1.395l.7-1.124zm2.81-1.066l2.071 2.095H24v-2.795h-.614v2.085l-2.062-2.085h-.795v2.795h.619zM0 13.393h.619v-2.095l2.076 2.095h.781v-2.795h-.619v2.085L.795 10.598H0zm4.843-2.795h.619v2.795h-.62zm4.486 2.204c-.02.005-.096.005-.124.005H6.743v.572h2.5c.019 0 .167 0 .195-.005.51-.048.743-.472.743-.843 0-.381-.243-.79-.705-.833-.09-.01-.166-.01-.2-.01H7.643a.83.83 0 0 1-.181-.014c-.129-.034-.176-.148-.176-.243 0-.086.047-.2.18-.238a.68.68 0 0 1 .172-.014h2.357v-.562H7.6c-.1 0-.176.004-.238.014a.792.792 0 0 0-.695.805c0 .343.214.743.685.81.086.009.205.009.258.009H9.2c.029 0 .1 0 .114.005.181.023.243.157.243.276a.262.262 0 0 1-.228.266zm4.657 0c-.02.005-.096.005-.129.005H11.4v.572h2.5c.019 0 .167 0 .195-.005.51-.048.743-.472.743-.843 0-.381-.243-.79-.705-.833-.09-.01-.166-.01-.2-.01H12.3a.83.83 0 0 1-.181-.014c-.129-.034-.176-.148-.176-.243 0-.086.047-.2.18-.238a.68.68 0 0 1 .172-.014h2.357v-.562h-2.395c-.1 0-.176.004-.238.014a.792.792 0 0 0-.695.805c0 .343.214.743.686.81.085.009.204.009.257.009h1.59c.029 0 .1 0 .114.005.181.023.243.157.243.276a.267.267 0 0 1-.228.266Z"/>
  </g>
</svg>""",

    "Skoda": """<svg viewBox="0 0 60 40" xmlns="http://www.w3.org/2000/svg">
  <rect width="60" height="40" fill="#186A3B"/>
  <g transform="translate(16.000,6.000) scale(1.16667)" fill="#FFFFFF">
    <path d="M12 0C5.3726 0 0 5.3726 0 12s5.3726 12 12 12 12-5.3726 12-12S18.6274 0 12 0Zm0 22.9636C5.945 22.9636 1.0364 18.055 1.0364 12 1.0364 5.945 5.945 1.0364 12 1.0364S22.9636 5.945 22.9636 12 18.055 22.9636 12 22.9636Zm5.189-7.2325-.269.7263h-.984c.263-.7089 3.5783-8.6177-2.9362-13.9819a9.5254 9.5254 0 0 0-4.0531.4483c.2172.175 2.474 2.0276 3.5373 4.315l-.312.084c-.5861-.6387-2.7156-2.9833-4.7448-3.7379a9.6184 9.6184 0 0 0-2.8448 2.3597c.953.4875 3.4432 1.9748 4.3896 3.1302-.0542.0244-.267.139-.267.139-1.736-1.3195-4.8199-2.0043-4.9775-2.0383a9.5126 9.5126 0 0 0-1.2267 3.6098c4.7759.9613 6.0618 3.1715 6.2818 5.6721H7.878l-1.5545-.6776a.8563.8563 0 0 0-.2524-.0531H3.1767a9.587 9.587 0 0 0 1.9267 2.9155h1.2334c.1063 0 .1993-.0133.2923-.0664l1.2489-.6378h9.042l.269.7264a4.8386 4.8386 0 0 0 2.9466-1.4667 4.839 4.839 0 0 0-2.9467-1.4666zm-4.14-.5786a1.1863 1.1863 0 0 1-.5038-1.2162 1.1862 1.1862 0 0 1 .931-.9309 1.1863 1.1863 0 0 1 1.2161.5038c.3098.4636.2563 1.0924-.1473 1.496-.4032.4032-1.0318.4574-1.496.1473z"/>
  </g>
</svg>""",

    "Chevrolet": """<svg viewBox="0 0 60 40" xmlns="http://www.w3.org/2000/svg">
  <rect width="60" height="40" fill="#0D0D0D"/>
  <g transform="translate(16.000,6.000) scale(1.16667)" fill="#D4AF37">
    <path d="M23.905 9.784H15.92V8.246a.157.157 0 00-.157-.158H8.238a.157.157 0 00-.157.158v1.538H2.358c-.087 0-.193.07-.237.158L.02 14.058c-.045.088-.011.157.077.157H8.08v1.54c0 .086.07.157.157.157h7.525c.087 0 .157-.07.157-.157v-1.54h5.723c.087 0 .193-.07.238-.157l2.1-4.116c.045-.087.011-.158-.076-.158m-2.494.996l-1.244 2.437h-5.232v1.708H9.07v-1.708H2.595L3.84 10.78h5.232V9.073h5.864v1.707z"/>
  </g>
</svg>""",

    "Fiat": """<svg viewBox="0 0 60 40" xmlns="http://www.w3.org/2000/svg">
  <rect width="60" height="40" fill="#FFFFFF"/>
  <g transform="translate(16.000,6.000) scale(1.16667)" fill="#E60026">
    <path d="M21.175 6.25c.489 1.148.726 2.442.726 3.956 0 .818-.068 1.69-.206 2.666-.286 2.01-1.048 4.11-1.75 5.494-.114.223-.205.371-.388.533-.32.282-.602.352-.706.291-.084-.05-.131-.302-.114-.673.014-.316.089-.55.204-.924a36.261 36.261 0 0 0 1.2-5.416c.385-2.664.37-5.06-.201-6.52a2.224 2.224 0 0 0-.22-.427c-.062-.09-.106-.136-.106-.136-1.181-1.183-4.37-1.776-7.56-1.776-3.19 0-6.378.593-7.558 1.776 0 0-.045.045-.106.136a2.122 2.122 0 0 0-.221.426c-.572 1.46-.586 3.857-.201 6.521.26 1.807.672 3.72 1.227 5.504.096.307.158.516.173.84.016.369-.03.62-.114.67-.104.06-.389-.01-.71-.295-.23-.205-.345-.405-.49-.701-.68-1.385-1.393-3.397-1.667-5.323a18.884 18.884 0 0 1-.206-2.666c0-1.514.238-2.807.726-3.954.367-.86.983-1.58 1.782-2.083a13.892 13.892 0 0 1 6.526-2.122 13.9 13.9 0 0 1 .815-.026h.02c.274 0 .548.01.818.026 2.282.138 4.539.873 6.525 2.122a4.583 4.583 0 0 1 1.782 2.082zm-4.763 14.526c-.088.019-.361.083-.632.157-.243.067-.483.12-.597.143a16.51 16.51 0 0 1-3.115.285h-.028c-1.117 0-2.177-.103-3.114-.285a9.23 9.23 0 0 1-.56-.133 14.987 14.987 0 0 0-.604-.148c-.418-.095-.796-.163-.817-.083-.025.093.162.288.401.472.056.042.195.14.357.22.15.073.32.128.386.15 1.098.355 2.346.502 3.941.502h.022c1.563 0 2.794-.142 3.877-.483.371-.117.59-.211.853-.42.22-.174.385-.353.361-.44-.02-.075-.348-.021-.731.063zm-2.508-10.313c-.145-.81-.32-1.432-.518-1.85l-.002-.004h-.021l-.682-.006h-.01l-.027 2.998h1.426l-.001-.01c0-.005-.056-.522-.165-1.128zm5.76 1.687c-.322 2.228-.88 4.623-1.66 6.701-.13.35-.248.48-.53.7a6.23 6.23 0 0 1-2.431 1.028c-.897.175-1.908.272-2.974.272h-.029a15.66 15.66 0 0 1-2.973-.272 6.23 6.23 0 0 1-2.433-1.028c-.282-.22-.399-.35-.527-.7-.782-2.078-1.34-4.473-1.661-6.701-.373-2.577-.35-4.847.18-6.202.067-.17.138-.292.19-.369.046-.065.078-.1.078-.1 1.068-1.07 4.06-1.652 7.16-1.652 3.101 0 6.093.582 7.16 1.653 0 0 .032.033.078.1.052.076.124.197.19.368.531 1.355.554 3.625.182 6.202zM8.904 7.565L6.222 7.55l.267 9.337 1.122.012-.016-4.25h1.014v-1.097H7.595V8.66h1.31V7.564zm1.876-.02l-1.365.003.181 9.35h1.157l.027-9.352zm3.448.014h-2.732l.108 9.334h1.092l.009-4.222h1.418l.002.007c.128.797.138 4.171.138 4.205v.015h1.063l.009-.479c.048-2.42.13-6.469-1.107-8.86zm4.32-.013l-3.344.013v1.077h.998v.01l-.042 8.252h1.132l.275-8.262h.981v-1.09zM23.975 12c0 6.617-5.372 12-11.976 12C5.397 24 .025 18.617.025 12S5.397 0 12 0c6.604 0 11.976 5.383 11.976 12zm-.33-.008C23.64 5.561 18.418.33 11.998.33 5.642.33.46 5.463.358 11.811a1.71 1.71 0 0 1 .684-.78c.655-.388.834-1.385.893-1.981l.012-.062c-.039.395-.07.79-.07 1.218 0 .832.07 1.718.21 2.708.412 2.9 1.813 6.007 2.637 6.958l.046.05.192.202.007.006c2.01 1.647 3.857 2.23 7.061 2.23h.022c3.203 0 5.05-.583 7.06-2.23l.009-.006.185-.197.052-.056c.826-.954 2.226-4.057 2.638-6.957.14-.99.209-1.876.209-2.708 0-.454-.021-.89-.064-1.309l.006.006c.06.597.379 2.141.995 2.586.21.152.375.317.503.503z"/>
  </g>
</svg>""",

    "Datsun": """<svg viewBox="0 0 60 40" xmlns="http://www.w3.org/2000/svg">
  <rect width="60" height="40" fill="#C41230"/>
  <path d="M5 33 Q5 7 30 7 Q55 7 55 33"
        fill="none" stroke="#FFFFFF" stroke-width="3.2" stroke-linecap="round"/>
  <path d="M11 33 Q11 14 30 14 Q49 14 49 33"
        fill="none" stroke="#FFFFFF" stroke-width="1.4"
        stroke-linecap="round" opacity="0.6"/>
  <text x="30" y="32" font-family="Arial Black,sans-serif"
        font-size="8.5" font-weight="900"
        fill="#FFFFFF" text-anchor="middle" letter-spacing="1.2">DATSUN</text>
</svg>""",

}


# ═══════════════════════════════════════════════════════════
# ENGINE SVG — proper cross-section schematic
# ═══════════════════════════════════════════════════════════
def classify_engine(cc, power, torque):
    if cc >= 2500 or power >= 180:
        return "performance", "High-Performance", "#F97316"
    elif torque >= 280:
        return "diesel_torque", "Diesel Torque", "#3B82F6"
    elif cc >= 1800:
        return "large", "Large Displacement", "#A78BFA"
    elif cc >= 1200:
        return "mid", "Mid-Range", "#34D399"
    else:
        return "small", "Small / Hatch", "#FBBF24"


def get_engine_svg(cc, power, torque):
    """
    Draws a realistic engine cross-section schematic.
    Uses layered transparency and gradients for depth:
      - dark bore walls with subtle oil-sheen highlight
      - piston crown with transparency so bore wall shows through edges
      - connecting rod as tapered shape with big-end bearing
      - crankshaft as actual circles
      - metal surfaces have subtle highlight edges
    """
    tier, tier_label, accent = classify_engine(cc, power, torque)

    if cc < 900:
        n = 3
    elif cc < 2200:
        n = 4
    else:
        n = 6

    VW = 340
    # Colors
    BG      = "#0D1117"   # near-black background
    BLOCK   = "#1E2A3A"   # engine block casing, dark blue-grey
    HEAD_C  = "#253040"   # cylinder head, slightly lighter
    BORE_C  = "#0F161E"   # bore interior, very dark
    SHINE1  = "#334155"   # metal highlight (lighter edge)
    SHINE2  = "#475569"   # secondary highlight
    DIMMED  = "#1A2535"   # recessed metal
    ROD_C   = "#3D5068"   # conrod
    CRANK_C = "#2A3A4E"   # crankshaft main body

    # Cylinder sizing
    if n == 6:
        CW = 30; CH = 62; GAP = 5
    elif n == 3:
        CW = 42; CH = 68; GAP = 10
    else:
        CW = 38; CH = 66; GAP = 8

    block_w  = n * CW + (n - 1) * GAP
    BPAD     = 10          # padding each side of block
    bx0      = (VW - block_w) / 2   # x of first cylinder

    # Y positions (all carefully stacked)
    CAM_Y    = 8           # camshaft top
    CAM_H    = 9
    SPARK_Y  = CAM_Y + CAM_H + 1   # spark plug tips
    SPARK_H  = 10
    HEAD_Y   = SPARK_Y + SPARK_H    # cylinder head top
    HEAD_H   = 18
    BY       = HEAD_Y + HEAD_H      # cylinder block top
    BORE_Y   = BY                   # same as block top
    CRANK_Y  = BY + CH + 4          # crankshaft centre
    SUMP_Y   = CRANK_Y + 14        # oil sump
    SUMP_H   = 12
    SPEC_Y   = SUMP_Y + SUMP_H + 16
    VH       = SPEC_Y + 18

    out = []

    # ── 1. background grid ─────────────────────────────────────
    out.append(f'<rect width="{VW}" height="{VH}" fill="{BG}"/>')
    out.append(
        f'<defs>'
        f'<pattern id="egrid" width="14" height="14" patternUnits="userSpaceOnUse">'
        f'<path d="M14 0L0 0 0 14" fill="none" stroke="#141E2B" stroke-width="0.5"/>'
        f'</pattern>'
        # Gradient for bore wall oil sheen (left = dark, right edge = faint highlight)
        f'<linearGradient id="bore_sheen" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0%" stop-color="#0A1018" stop-opacity="1"/>'
        f'<stop offset="75%" stop-color="#0F161E" stop-opacity="1"/>'
        f'<stop offset="100%" stop-color="#1E2D3D" stop-opacity="0.6"/>'
        f'</linearGradient>'
        # Gradient for piston top face (crown highlight)
        f'<linearGradient id="piston_top" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{accent}" stop-opacity="1"/>'
        f'<stop offset="100%" stop-color="{accent}" stop-opacity="0.55"/>'
        f'</linearGradient>'
        # Gradient for metal block top edge (bevel)
        f'<linearGradient id="metal_bevel" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{SHINE2}" stop-opacity="0.9"/>'
        f'<stop offset="100%" stop-color="{BLOCK}" stop-opacity="1"/>'
        f'</linearGradient>'
        f'<linearGradient id="crank_grad" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{SHINE2}"/>'
        f'<stop offset="100%" stop-color="{CRANK_C}"/>'
        f'</linearGradient>'
        f'<marker id="earrow" viewBox="0 0 10 10" refX="8" refY="5"'
        f' markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
        f'<path d="M2 2L8 5L2 8" fill="none" stroke="#60A5FA"'
        f' stroke-width="1.5" stroke-linecap="round"/></marker>'
        f'<marker id="earrow2" viewBox="0 0 10 10" refX="8" refY="5"'
        f' markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
        f'<path d="M2 2L8 5L2 8" fill="none" stroke="#FC8181"'
        f' stroke-width="1.5" stroke-linecap="round"/></marker>'
        f'</defs>'
    )
    out.append(f'<rect width="{VW}" height="{VH}" fill="url(#egrid)"/>')

    # ── 2. Camshaft rail ───────────────────────────────────────
    cx0 = bx0 - BPAD + 4
    cw  = block_w + 2*BPAD - 8
    out.append(
        f'<rect x="{cx0}" y="{CAM_Y}" width="{cw}" height="{CAM_H}"'
        f' rx="4" fill="{HEAD_C}" stroke="{SHINE1}" stroke-width="1"/>'
    )
    # cam lobes — small ellipses per cylinder, accent-colored tip
    for i in range(n):
        lx = bx0 + i*(CW+GAP) + CW/2
        out.append(
            f'<ellipse cx="{lx:.1f}" cy="{CAM_Y+4}" rx="4.5" ry="6"'
            f' fill="{DIMMED}" stroke="{SHINE2}" stroke-width="0.8"/>'
            f'<ellipse cx="{lx:.1f}" cy="{CAM_Y+1}" rx="2" ry="2"'
            f' fill="{accent}" opacity="0.7"/>'
        )

    # ── 3. Spark plugs ─────────────────────────────────────────
    for i in range(n):
        lx = bx0 + i*(CW+GAP) + CW/2
        # Ceramic body
        out.append(
            f'<rect x="{lx-2.5:.1f}" y="{SPARK_Y}" width="5" height="{SPARK_H}"'
            f' rx="1.5" fill="{SHINE2}" stroke="{SHINE1}" stroke-width="0.6"/>'
        )
        # Electrode tip glow
        out.append(
            f'<circle cx="{lx:.1f}" cy="{SPARK_Y+SPARK_H-1}" r="2"'
            f' fill="{accent}" opacity="0.85"/>'
        )

    # ── 4. Cylinder head block ──────────────────────────────────
    out.append(
        f'<rect x="{bx0-BPAD}" y="{HEAD_Y}" width="{block_w+2*BPAD}" height="{HEAD_H}"'
        f' rx="4" fill="{HEAD_C}" stroke="{SHINE2}" stroke-width="1"/>'
        # top highlight bevel
        f'<rect x="{bx0-BPAD}" y="{HEAD_Y}" width="{block_w+2*BPAD}" height="3"'
        f' rx="2" fill="url(#metal_bevel)" opacity="0.5"/>'
    )
    # valve ports (intake+exhaust per cylinder visible as dark ovals in head)
    for i in range(n):
        lx = bx0 + i*(CW+GAP) + CW/2
        out.append(
            f'<ellipse cx="{lx-4:.1f}" cy="{HEAD_Y+HEAD_H-3}" rx="3" ry="2"'
            f' fill="{BORE_C}" stroke="#60A5FA" stroke-width="0.7" opacity="0.8"/>'
            f'<ellipse cx="{lx+4:.1f}" cy="{HEAD_Y+HEAD_H-3}" rx="3" ry="2"'
            f' fill="{BORE_C}" stroke="#FC8181" stroke-width="0.7" opacity="0.8"/>'
        )

    # ── 5. Engine block outer casing ───────────────────────────
    out.append(
        f'<rect x="{bx0-BPAD}" y="{BY}" width="{block_w+2*BPAD}" height="{CH+18}"'
        f' rx="4" fill="{BLOCK}" stroke="{SHINE1}" stroke-width="1"/>'
        # top edge bevel
        f'<rect x="{bx0-BPAD}" y="{BY}" width="{block_w+2*BPAD}" height="3"'
        f' fill="{SHINE2}" opacity="0.35" rx="2"/>'
    )

    # ── 6. Individual cylinders ────────────────────────────────
    for i in range(n):
        cx  = bx0 + i*(CW+GAP)
        bx  = cx + 3          # bore x start
        bw  = CW - 6          # bore width
        pyo = 22              # piston y-offset from bore top

        # Bore wall (with sheen gradient)
        out.append(
            f'<rect x="{bx:.1f}" y="{BORE_Y}" width="{bw:.1f}" height="{CH}"'
            f' rx="2" fill="url(#bore_sheen)" stroke="{SHINE1}" stroke-width="0.6"/>'
        )
        # Bore wall left edge highlight
        out.append(
            f'<line x1="{bx:.1f}" y1="{BORE_Y+4}" x2="{bx:.1f}" y2="{BORE_Y+CH-4}"'
            f' stroke="{SHINE2}" stroke-width="0.8" opacity="0.4"/>'
        )

        # Piston body (semi-transparent so bore edges visible)
        px  = bx + 1
        py  = BORE_Y + pyo
        pw  = bw - 2
        ph  = 16
        out.append(
            f'<rect x="{px:.1f}" y="{py}" width="{pw:.1f}" height="{ph}"'
            f' rx="2" fill="url(#piston_top)" opacity="0.92"/>'
        )
        # Piston crown dome (subtle arch)
        out.append(
            f'<path d="M{px:.1f},{py+2} Q{px+pw/2:.1f},{py-3} {px+pw:.1f},{py+2}"'
            f' fill="{accent}" opacity="0.3" stroke="none"/>'
        )
        # Piston rings (3 rings, look like real cross-sections)
        for ri, ry_off in enumerate([1, 5, 10]):
            opacity = 0.5 if ri < 2 else 0.3
            out.append(
                f'<rect x="{bx:.1f}" y="{py+ry_off}" width="2.5" height="2.5"'
                f' fill="{SHINE2}" opacity="{opacity}"/>'
                f'<rect x="{bx+bw-2.5:.1f}" y="{py+ry_off}" width="2.5" height="2.5"'
                f' fill="{SHINE2}" opacity="{opacity}"/>'
            )

        # Connecting rod – tapered shape (wider at top pin, narrower in middle, big-end circle)
        rod_cx = cx + CW/2
        rod_top_y = py + ph
        rod_bot_y = CRANK_Y
        rod_bot_y_contact = CRANK_Y - 8
        # Rod body (tapered path)
        tw = 3.5   # half-width at top
        bw2 = 2.5  # half-width at bottom
        out.append(
            f'<path d="M{rod_cx-tw:.1f},{rod_top_y} '
            f'L{rod_cx-bw2:.1f},{rod_bot_y_contact} '
            f'L{rod_cx+bw2:.1f},{rod_bot_y_contact} '
            f'L{rod_cx+tw:.1f},{rod_top_y} Z"'
            f' fill="{ROD_C}" stroke="{SHINE1}" stroke-width="0.6"/>'
        )
        # Big-end bearing circle at crankshaft
        out.append(
            f'<circle cx="{rod_cx:.1f}" cy="{CRANK_Y}" r="7"'
            f' fill="{CRANK_C}" stroke="{SHINE2}" stroke-width="0.8"/>'
            f'<circle cx="{rod_cx:.1f}" cy="{CRANK_Y}" r="4"'
            f' fill="{BG}" stroke="{SHINE1}" stroke-width="0.6"/>'
        )

    # ── 7. Crankshaft main journal ──────────────────────────────
    crank_x0 = bx0 - BPAD + 4
    crank_len = block_w + 2*BPAD - 8
    out.append(
        f'<rect x="{crank_x0:.1f}" y="{CRANK_Y-5}" width="{crank_len:.1f}" height="10"'
        f' rx="5" fill="url(#crank_grad)" stroke="{SHINE2}" stroke-width="0.8"/>'
    )
    # Main bearing circles
    for i in range(n + 1):
        mx = bx0 + i*(CW+GAP) - GAP/2
        if i == 0:
            mx = bx0 - BPAD/2
        elif i == n:
            mx = bx0 + block_w + BPAD/2
        out.append(
            f'<circle cx="{mx:.1f}" cy="{CRANK_Y}" r="5"'
            f' fill="{CRANK_C}" stroke="{SHINE2}" stroke-width="0.8"/>'
            f'<circle cx="{mx:.1f}" cy="{CRANK_Y}" r="2.5"'
            f' fill="{BG}" stroke="{SHINE1}" stroke-width="0.5"/>'
        )

    # ── 8. Oil sump ─────────────────────────────────────────────
    sx0 = bx0 - BPAD + 2
    sw  = block_w + 2*BPAD - 4
    out.append(
        f'<path d="M{sx0:.1f},{SUMP_Y} '
        f'L{sx0:.1f},{SUMP_Y+SUMP_H-4} '
        f'Q{sx0:.1f},{SUMP_Y+SUMP_H} {sx0+6:.1f},{SUMP_Y+SUMP_H} '
        f'L{sx0+sw-6:.1f},{SUMP_Y+SUMP_H} '
        f'Q{sx0+sw:.1f},{SUMP_Y+SUMP_H} {sx0+sw:.1f},{SUMP_Y+SUMP_H-4} '
        f'L{sx0+sw:.1f},{SUMP_Y} Z"'
        f' fill="{BLOCK}" stroke="{SHINE1}" stroke-width="0.8" opacity="0.85"/>'
    )
    # Oil level line
    out.append(
        f'<line x1="{sx0+6:.1f}" y1="{SUMP_Y+7}" x2="{sx0+sw-6:.1f}" y2="{SUMP_Y+7}"'
        f' stroke="#3B6EA0" stroke-width="1" stroke-dasharray="3,3" opacity="0.5"/>'
    )

    # ── 9. Timing chain (right side, compact) ──────────────────
    tc_x = min(bx0 + block_w + BPAD + 5, VW - 28)
    tc_cx = tc_x + 9
    if tc_cx + 11 < VW:
        out.append(
            f'<circle cx="{tc_cx:.1f}" cy="{CAM_Y+5}" r="7"'
            f' fill="none" stroke="{accent}" stroke-width="1.2"'
            f' stroke-dasharray="2.5,2" opacity="0.7"/>'
            f'<circle cx="{tc_cx:.1f}" cy="{CRANK_Y}" r="10"'
            f' fill="none" stroke="{accent}" stroke-width="1.2"'
            f' stroke-dasharray="2.5,2" opacity="0.7"/>'
            f'<line x1="{tc_cx-7:.1f}" y1="{CAM_Y+5}" x2="{tc_cx-10:.1f}" y2="{CRANK_Y}"'
            f' stroke="{accent}" stroke-width="1.2" opacity="0.45"/>'
            f'<line x1="{tc_cx+7:.1f}" y1="{CAM_Y+5}" x2="{tc_cx+10:.1f}" y2="{CRANK_Y}"'
            f' stroke="{accent}" stroke-width="1.2" opacity="0.45"/>'
        )

    # ── 10. Intake/exhaust arrows on first 2 cylinders ─────────
    for i in range(min(2, n)):
        lx = bx0 + i*(CW+GAP) + CW/2
        # Intake arrow (down, blue)
        arrow_top = HEAD_Y - 12
        out.append(
            f'<line x1="{lx-5:.1f}" y1="{arrow_top}" x2="{lx-5:.1f}" y2="{HEAD_Y-1}"'
            f' stroke="#60A5FA" stroke-width="1.8" stroke-linecap="round"'
            f' marker-end="url(#earrow)"/>'
        )
        # Exhaust arrow (up, red)
        out.append(
            f'<line x1="{lx+5:.1f}" y1="{HEAD_Y-1}" x2="{lx+5:.1f}" y2="{arrow_top}"'
            f' stroke="#FC8181" stroke-width="1.8" stroke-linecap="round"'
            f' marker-end="url(#earrow2)"/>'
        )

    # ── 11. Tier badge (top-left) ──────────────────────────────
    out.append(
        f'<rect x="5" y="5" width="90" height="16" rx="8"'
        f' fill="{accent}" fill-opacity="0.12" stroke="{accent}" stroke-width="0.7" stroke-opacity="0.5"/>'
        f'<text x="50" y="16" font-family="monospace" font-size="8"'
        f' fill="{accent}" text-anchor="middle" font-weight="bold"'
        f' letter-spacing="0.3">{tier_label.upper()}</text>'
    )

    # ── 12. Spec label ─────────────────────────────────────────
    out.append(
        f'<text x="{VW/2}" y="{SPEC_Y}"'
        f' font-family="monospace" font-size="9.5" fill="{accent}"'
        f' text-anchor="middle" font-weight="bold">'
        f'{cc:,} cc  ·  {power:.0f} bhp  ·  {torque:.0f} Nm</text>'
        f'<text x="{VW/2}" y="{SPEC_Y+12}"'
        f' font-family="monospace" font-size="7.5" fill="{SHINE2}"'
        f' text-anchor="middle">{n}-Cylinder  ·  {tier_label}</text>'
    )

    # ── 13. Legend ─────────────────────────────────────────────
    lx_leg = VW - 85
    ly_leg = SPEC_Y + 4
    out.append(
        f'<circle cx="{lx_leg}" cy="{ly_leg}" r="4" fill="#60A5FA"/>'
        f'<text x="{lx_leg+7}" y="{ly_leg+4}" font-family="monospace" font-size="7" fill="{SHINE2}">intake</text>'
        f'<circle cx="{lx_leg+46}" cy="{ly_leg}" r="4" fill="#FC8181"/>'
        f'<text x="{lx_leg+53}" y="{ly_leg+4}" font-family="monospace" font-size="7" fill="{SHINE2}">exh</text>'
    )

    svg = (
        f'<svg viewBox="0 0 {VW} {VH}" xmlns="http://www.w3.org/2000/svg"'
        f' style="width:100%;height:auto;display:block;">'
        + ''.join(out) +
        f'</svg>'
    )
    return svg, tier_label, accent


# ═══════════════════════════════════════════════════════════
# CSS + THEME TOGGLE + HEADER
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg-page:#F4F6FA;--bg-card:#FFFFFF;--bg-input:#FFFFFF;--bg-engine:#F8FAFC;
    --border:#D8E2F0;--border-hover:#94B4E8;
    --text-primary:#0F1929;--text-muted:#5A6A82;--text-faint:#8CA0BC;
    --accent:#1E5FD4;--accent-muted:#4F8EF7;--divider:#CBD8EE;
    --badge-bg:#EBF1FF;--badge-border:#C0D4F8;--badge-text:#1E5FD4;
    --pred-bg1:#EBF1FF;--pred-bg2:#DCE9FF;--pred-border:#94B4E8;
    --pred-label:#3B72D4;--pred-price:#0F1929;--pred-sub:#5A6A82;
    --fi-bg:#D8E2F0;--fi-c1:#3B72D4;--fi-c2:#64AFFF;
    --chip-bg:#EBF1FF;--chip-bd:#C0D4F8;--chip-tx:#5A6A82;--chip-v:#0F1929;
    --met-bg:#FFFFFF;--met-bd:#D8E2F0;--met-lbl:#5A6A82;--met-v:#0F1929;
    --sec-col:#1E5FD4;--hr:#D8E2F0;--ft:#8CA0BC;
    --eng-bd:#D8E2F0;--eng-hd:#F0F4FB;
    --brand-bg:#FFFFFF;--brand-bd:#D8E2F0;--brand-sub:#8CA0BC;
    --sel-bg:#FFFFFF;--sel-bd:#C0D4F8;--inp-col:#0F1929;
    --trk:#CBD8EE;--thm:#FFFFFF;--thm-sh:rgba(0,0,0,.15);
    --hglow:rgba(30,95,212,.07);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg-page:#0A0E1A;--bg-card:#111827;--bg-input:#151E2E;--bg-engine:#0D1117;
    --border:#1E2A3E;--border-hover:#2A4A8A;
    --text-primary:#E2E8F0;--text-muted:#8892A4;--text-faint:#4F6A8E;
    --accent:#4F8EF7;--accent-muted:#60A5FA;--divider:#1E2A3E;
    --badge-bg:rgba(79,142,247,.12);--badge-border:rgba(79,142,247,.28);--badge-text:#7AABF8;
    --pred-bg1:#0F2044;--pred-bg2:#162856;--pred-border:#2A4A8A;
    --pred-label:#5B8EE6;--pred-price:#FFFFFF;--pred-sub:#6B7A96;
    --fi-bg:#1A2332;--fi-c1:#3B6CD4;--fi-c2:#64AFFF;
    --chip-bg:#1A2332;--chip-bd:#2A3A52;--chip-tx:#8892A4;--chip-v:#CBD5E1;
    --met-bg:#111827;--met-bd:#1E2A3E;--met-lbl:#6B7A96;--met-v:#FFFFFF;
    --sec-col:#4F8EF7;--hr:#1E2A3E;--ft:#3D4F6A;
    --eng-bd:#1E2A3E;--eng-hd:#111827;
    --brand-bg:#111827;--brand-bd:#1E2A3E;--brand-sub:#4F6A8E;
    --sel-bg:#151E2E;--sel-bd:#2A3A52;--inp-col:#E2E8F0;
    --trk:#2A4A8A;--thm:#4F8EF7;--thm-sh:rgba(79,142,247,.3);
    --hglow:rgba(79,142,247,.10);
  }
}

*,*::before,*::after{transition:background-color .3s,border-color .3s,color .3s,box-shadow .3s;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;background-color:var(--bg-page) !important;}
.main,.stApp{background-color:var(--bg-page) !important;}
.block-container{padding:0 1.5rem 3rem;max-width:1100px;}

/* ── Dark mode toggle ── */

/* ── Header ── */
.app-header{position:relative;text-align:center;padding:2rem 2rem 1.6rem;margin-bottom:.5rem;overflow:hidden;}
.app-header::before{content:'';position:absolute;top:0;left:50%;transform:translateX(-50%);width:600px;height:160px;background:radial-gradient(ellipse at center,var(--hglow) 0%,transparent 70%);pointer-events:none;}
.app-header h1{font-family:'Space Grotesk',sans-serif;font-size:2.8rem;font-weight:700;letter-spacing:-.035em;line-height:1.1;margin:0 0 .5rem;color:var(--text-primary);}
.app-header h1 .gt{background:linear-gradient(135deg,#2563EB 0%,#3B82F6 50%,#60A5FA 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hdr-rule{width:44px;height:3px;background:linear-gradient(90deg,#2563EB,#60A5FA);border-radius:2px;margin:.75rem auto .85rem;}
.hdr-sub{display:flex;align-items:center;justify-content:center;gap:.9rem;font-size:.81rem;color:var(--text-muted);flex-wrap:wrap;}
.hdr-sub .sep{opacity:.3;}
.cd-badge{display:inline-flex;align-items:center;gap:.3rem;background:var(--badge-bg);border:1px solid var(--badge-border);border-radius:6px;padding:.18rem .6rem;font-size:.72rem;font-weight:600;color:var(--badge-text);letter-spacing:.03em;}

/* ── Section labels ── */
.section-label{font-size:.67rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--sec-col);margin-bottom:.8rem;padding-bottom:.4rem;border-bottom:1.5px solid var(--divider);}

/* ── Brand card ── */
.brand-display{display:flex;align-items:center;gap:.85rem;background:var(--brand-bg);border:1.5px solid var(--brand-bd);border-radius:12px;padding:.6rem 1rem;margin-top:.5rem;min-height:58px;}
.brand-display:hover{border-color:var(--border-hover);box-shadow:0 2px 14px rgba(79,142,247,.12);}
.brand-logo-wrap{width:62px;height:42px;flex-shrink:0;display:flex;align-items:center;justify-content:center;border-radius:8px;overflow:hidden;box-shadow:0 1px 5px rgba(0,0,0,.12);}
.brand-logo-wrap svg{width:100%;height:100%;}
.brand-name-block{flex:1;}
.brand-name-block .bname{font-family:'Space Grotesk',sans-serif;font-size:.95rem;font-weight:600;color:var(--text-primary);}
.brand-name-block .bsub{font-size:.7rem;color:var(--brand-sub);margin-top:.1rem;}

/* ── Engine card ── */
.engine-card{background:var(--bg-engine);border:1.5px solid var(--eng-bd);border-radius:14px;overflow:hidden;margin-top:.8rem;}
.engine-card-header{display:flex;align-items:center;gap:.5rem;padding:.5rem 1rem;border-bottom:1px solid var(--eng-bd);background:var(--eng-hd);}
.engine-card-header .etitle{font-size:.7rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;}
.engine-legend{display:flex;gap:1rem;font-size:.69rem;color:var(--text-faint);margin-left:auto;align-items:center;}
.engine-legend .li{display:flex;align-items:center;gap:.3rem;}
.engine-legend .dot{width:7px;height:7px;border-radius:50%;}
.engine-card-body{padding:.4rem .4rem 0;}

/* ── Prediction banner ── */
.pred-banner{background:linear-gradient(135deg,var(--pred-bg1) 0%,var(--pred-bg2) 100%);border:1.5px solid var(--pred-border);border-radius:16px;padding:1.8rem 2rem;text-align:center;margin-top:1.5rem;}
.pred-banner .label{font-size:.72rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--pred-label);margin-bottom:.45rem;}
.pred-banner .price{font-family:'Space Grotesk',sans-serif;font-size:2.8rem;font-weight:700;color:var(--pred-price);letter-spacing:-.03em;}
.pred-banner .subtext{font-size:.82rem;color:var(--pred-sub);margin-top:.35rem;}

/* ── Feature importance ── */
.fi-row{display:flex;align-items:center;margin-bottom:.5rem;gap:.7rem;}
.fi-label{font-size:.77rem;color:var(--text-muted);width:110px;flex-shrink:0;text-align:right;}
.fi-bar-bg{flex:1;background:var(--fi-bg);border-radius:4px;height:7px;overflow:hidden;}
.fi-bar-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--fi-c1),var(--fi-c2));}
.fi-pct{font-size:.71rem;color:var(--accent);width:38px;text-align:right;font-weight:600;}

/* ── Chips ── */
.chip-row{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.9rem;}
.chip{background:var(--chip-bg);border:1px solid var(--chip-bd);border-radius:20px;padding:.22rem .7rem;font-size:.71rem;color:var(--chip-tx);}
.chip span{color:var(--chip-v);font-weight:600;}

/* ── Streamlit overrides ── */
div[data-testid="stSlider"]>label,div[data-testid="stNumberInput"]>label,div[data-testid="stSelectbox"]>label{font-size:.79rem !important;font-weight:500 !important;color:var(--text-muted) !important;margin-bottom:.2rem !important;}
.stButton>button{background:linear-gradient(135deg,#1E4FC8,#4F8EF7) !important;color:white !important;border:none !important;border-radius:10px !important;padding:.65rem 2rem !important;font-size:.88rem !important;font-weight:600 !important;font-family:'Inter',sans-serif !important;letter-spacing:.025em !important;width:100% !important;box-shadow:0 2px 12px rgba(79,142,247,.25) !important;}
.stButton>button:hover{opacity:.9 !important;transform:translateY(-1px) !important;}
.stButton>button:active{transform:translateY(0) !important;}
.stSelectbox [data-baseweb="select"]{background-color:var(--sel-bg) !important;border-color:var(--sel-bd) !important;}
.stNumberInput input{background-color:var(--bg-input) !important;border-color:var(--sel-bd) !important;color:var(--inp-col) !important;}
div[data-testid="metric-container"]{background:var(--met-bg) !important;border:1.5px solid var(--met-bd) !important;border-radius:12px !important;padding:.9rem !important;}
div[data-testid="metric-container"] label{color:var(--met-lbl) !important;font-size:.72rem !important;letter-spacing:.05em !important;text-transform:uppercase !important;}
div[data-testid="metric-container"] [data-testid="stMetricValue"]{color:var(--met-v) !important;font-family:'Space Grotesk',sans-serif !important;font-size:1.35rem !important;}
hr{border-color:var(--hr) !important;}
</style>
""", unsafe_allow_html=True)

# ── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header"><h1>Car Resale <span class="gt">Price Predictor</span></h1><div class="hdr-rule"></div><div class="hdr-sub"><span>XGBoost ML Model</span><span class="sep">&#xB7;</span><span>32 Features</span><span class="sep">&#xB7;</span><span>15 Indian Brands</span><span class="sep">&#xB7;</span><span class="cd-badge">Trained on CarDekho's resale data</span></div></div>
""", unsafe_allow_html=True)
# ── LAYOUT ──────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:

    # Vehicle Identity
    st.markdown('<div class="section-label">Vehicle Identity</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        brand = st.selectbox("Brand", brands,
                             index=brands.index("Maruti") if "Maruti" in brands else 0)
        logo_svg = BRAND_SVG.get(brand, (
            f'<svg viewBox="0 0 60 40" xmlns="http://www.w3.org/2000/svg">'
            f'<rect width="60" height="40" fill="#1A2332"/>'
            f'<text x="30" y="24" font-family="Arial,sans-serif" font-size="9" '
            f'fill="#8892A4" text-anchor="middle">{brand[:8]}</text></svg>'
        ))
        st.markdown(f"""
        <div class="brand-display">
          <div class="brand-logo-wrap">{logo_svg}</div>
          <div class="brand-name-block">
            <div class="bname">{brand}</div>
            <div class="bsub">Selected brand</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        year = st.slider("Model Year", 2000, 2025, 2018)

    c3, c4 = st.columns(2)
    with c3:
        fuel = st.selectbox("Fuel Type", fuel_types,
                            index=fuel_types.index("Petrol") if "Petrol" in fuel_types else 0)
    with c4:
        transmission = st.selectbox("Transmission", trans_types,
                                    index=trans_types.index("Manual") if "Manual" in trans_types else 0)

    st.markdown("<br>", unsafe_allow_html=True)

    # Performance & Build
    st.markdown('<div class="section-label">Performance & Build</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        engine    = st.number_input("Engine (cc)",     min_value=500,  max_value=6000, value=1200, step=100)
        max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=500.0, value=82.0, step=1.0, format="%.1f")
    with c6:
        torque  = st.number_input("Torque (Nm)",   min_value=30.0, max_value=800.0, value=113.0, step=1.0, format="%.1f")
        mileage = st.number_input("Mileage (kmpl)", min_value=5.0,  max_value=45.0,  value=21.4, step=0.1, format="%.1f")

    # Engine diagram
    eng_svg, eng_label, eng_accent = get_engine_svg(engine, max_power, torque)
    st.markdown(f"""
    <div class="engine-card">
      <div class="engine-card-header">
        <span style="width:8px;height:8px;border-radius:50%;
                     background:{eng_accent};display:inline-block;flex-shrink:0;"></span>
        <span class="etitle" style="color:{eng_accent};">{eng_label} Engine</span>
        <div class="engine-legend">
          <div class="li"><div class="dot" style="background:#60A5FA;"></div>intake</div>
          <div class="li"><div class="dot" style="background:#FC8181;"></div>exhaust</div>
          <div class="li"><div class="dot" style="background:{eng_accent};"></div>piston</div>
        </div>
      </div>
      <div class="engine-card-body">{eng_svg}</div>
    </div>
    """, unsafe_allow_html=True)

    seats = st.select_slider("Seats", options=[2, 4, 5, 6, 7, 8, 9], value=5)

    st.markdown("<br>", unsafe_allow_html=True)

    # Ownership History
    st.markdown('<div class="section-label">Ownership History</div>', unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    with c7:
        km_driven = st.number_input("KM Driven", min_value=0, max_value=500000, value=45000, step=1000)
        owner     = st.selectbox("Owner History", owner_types)
    with c8:
        seller_type = st.selectbox("Seller Type", seller_types,
                                   index=seller_types.index("Individual") if "Individual" in seller_types else 0)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Predict Resale Price →")


# ── RIGHT COLUMN ────────────────────────────────────────────
with right_col:
    car_age = 2024 - year

    if predict_btn:
        input_dict = {
            'year': year, 'km_driven': km_driven, 'mileage': mileage,
            'engine': engine, 'max_power': max_power, 'torque': torque,
            'seats': seats, 'car_age': car_age,
        }
        input_dict[f"fuel_{fuel}"]                = 1
        input_dict[f"transmission_{transmission}"] = 1
        input_dict[f"seller_type_{seller_type}"]   = 1
        if owner != "First Owner":
            input_dict[f"owner_{owner}"]           = 1
        input_dict[f"brand_{brand}"]               = 1

        input_df = pd.DataFrame([input_dict]).reindex(columns=columns, fill_value=0)

        try:
            try:
                prediction = xgb_pipeline.predict(input_df)[0]
            except Exception:
                prediction = model.predict(input_df)[0]
            prediction = max(prediction, 0)

            st.markdown(f"""
            <div class="pred-banner">
              <div class="label">Estimated Resale Price</div>
              <div class="price">₹ {prediction:,.0f}</div>
              <div class="subtext">{brand} · {year} · {fuel} · {transmission}</div>
            </div>
            """, unsafe_allow_html=True)

            low, high = prediction * 0.92, prediction * 1.08
            st.markdown("<br>", unsafe_allow_html=True)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Lower Bound", f"₹{low:,.0f}")
            mc2.metric("Estimated",   f"₹{prediction:,.0f}")
            mc3.metric("Upper Bound", f"₹{high:,.0f}")

            km_per_yr = km_driven / max(car_age, 1)
            st.markdown(f"""
            <div class="chip-row">
              <div class="chip">Age <span>{car_age} yr{'s' if car_age != 1 else ''}</span></div>
              <div class="chip">Avg usage <span>{km_per_yr:,.0f} km/yr</span></div>
              <div class="chip">Seats <span>{seats}</span></div>
              <div class="chip">Owner <span>{owner}</span></div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    else:
        st.markdown("""
        <div class="pred-banner" style="opacity:0.55;">
          <div class="label">Awaiting Input</div>
          <div class="price" style="font-size:2rem;color:#5B7BAD;">₹ ——</div>
          <div class="subtext">Fill in the form and click Predict</div>
        </div>
        """, unsafe_allow_html=True)

    # Feature Importance
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">What drives the price</div>', unsafe_allow_html=True)
    try:
        xgb_step = xgb_pipeline.named_steps['model']
        fi_vals  = xgb_step.feature_importances_
        fi_names = xgb_step.feature_names_in_
    except Exception:
        fi_vals  = model.feature_importances_ if hasattr(model, 'feature_importances_') else model[-1].feature_importances_
        fi_names = columns

    fi_pairs = sorted(zip(fi_names, fi_vals), key=lambda x: -x[1])[:8]
    max_fi   = fi_pairs[0][1]
    fi_html  = ""
    for name, score in fi_pairs:
        lbl     = name.split("_", 1)[-1].replace("_", " ").title()
        bar_pct = int((score / max_fi) * 100)
        fi_html += (
            f'<div class="fi-row">'
            f'<div class="fi-label">{lbl}</div>'
            f'<div class="fi-bar-bg">'
            f'<div class="fi-bar-fill" style="width:{bar_pct}%"></div></div>'
            f'<div class="fi-pct">{score*100:.1f}%</div></div>'
        )
    st.markdown(fi_html, unsafe_allow_html=True)

    # Model Details
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Model Details</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="chip-row">
      <div class="chip">Algorithm <span>XGBoost</span></div>
      <div class="chip">Estimators <span>400</span></div>
      <div class="chip">Max Depth <span>6</span></div>
      <div class="chip">LR <span>0.05</span></div>
      <div class="chip">Features <span>32</span></div>
      <div class="chip">Brands <span>15</span></div>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ──────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#3D4F6A;font-size:0.75rem;'>"
    "Predictions are estimates based on historical data · Always verify with a certified valuation"
    "</p>",
    unsafe_allow_html=True
)