import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import glob
import os

# -----------------------------
# Constants
# -----------------------------
FT_TO_M = 0.3048
FTS_TO_MPS = 0.3048
FTS_TO_MPH = 0.681818
FT_TO_IN = 12.0

WINDOW_LENGTH = 11
POLYORDER = 2

DIST_X = np.sqrt(25.77*25.77 + 8.06*8.05)
DIST_Y = np.sqrt(DIST_X**2 + 1.87**2)

# -----------------------------
# Processing function (cached)
# -----------------------------
@st.cache_data
def process_dat_file(file_path):
    """Read a .dat file (whitespace-delimited) and compute all metrics."""
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    data = data.apply(pd.to_numeric, errors="coerce")
    
    time = data.iloc[:, 0].values
    hip_y_abs = data.iloc[:, 2].values
    wrist_x = data.iloc[:, 3].values
    wrist_y = data.iloc[:, 4].values
    
    hip_y_smooth = savgol_filter(hip_y_abs, WINDOW_LENGTH, POLYORDER)
    hip_y_rel = hip_y_abs - hip_y_abs[0]
    dt = np.mean(np.diff(time))
    
    trophy_idx = np.argmin(hip_y_smooth)
    trophy_time = time[trophy_idx]
    trophy_y = hip_y_smooth[trophy_idx]
    
    contact_idx = np.argmin(np.abs(time - 0.0))
    contact_time = time[contact_idx]
    contact_y = hip_y_smooth[contact_idx]
    
    # Displacement (trophy → contact)
    y_extension_ft = contact_y - trophy_y
    y_extension_m = y_extension_ft * FT_TO_M
    y_extension_in = y_extension_ft * FT_TO_IN
    
    hip_vy_fts = savgol_filter(hip_y_rel, WINDOW_LENGTH, POLYORDER, deriv=1, delta=dt)
    
    wrist_x_smooth = savgol_filter(wrist_x, WINDOW_LENGTH, POLYORDER)
    wrist_y_smooth = savgol_filter(wrist_y, WINDOW_LENGTH, POLYORDER)
    wrist_vx_fts = savgol_filter(wrist_x_smooth, WINDOW_LENGTH, POLYORDER, deriv=1, delta=dt)
    wrist_vy_fts = savgol_filter(wrist_y_smooth, WINDOW_LENGTH, POLYORDER, deriv=1, delta=dt)
    wrist_speed_fts = np.sqrt(wrist_vx_fts**2 + wrist_vy_fts**2)
    
    serve_kmh = (DIST_Y * 3600) / (time[-1] * 1000)
    serve_mph = serve_kmh * 0.621371
    
    peak_hip_vy_fts = np.max(np.abs(hip_vy_fts))
    peak_hip_vy_idx = np.argmax(np.abs(hip_vy_fts))
    peak_hip_vy_time = time[peak_hip_vy_idx]
    # Convert peak hip velocity
    peak_hip_vy_mps = peak_hip_vy_fts * FTS_TO_MPS
    peak_hip_vy_mph = peak_hip_vy_fts * FTS_TO_MPH
    
    peak_wrist_speed_fts = np.max(wrist_speed_fts)
    peak_wrist_speed_idx = np.argmax(wrist_speed_fts)
    peak_wrist_speed_time = time[peak_wrist_speed_idx]
    # Convert peak wrist speed
    peak_wrist_speed_mps = peak_wrist_speed_fts * FTS_TO_MPS
    peak_wrist_speed_mph = peak_wrist_speed_fts * FTS_TO_MPH
    
    return {
        'player_name': os.path.basename(file_path).replace('.dat', ''),
        'time': time,
        'hip_y_smooth': hip_y_smooth,
        'hip_vy_fts': hip_vy_fts,
        'wrist_speed_fts': wrist_speed_fts,
        'trophy_time': trophy_time,
        'trophy_y': trophy_y,
        'contact_time': contact_time,
        'contact_y': contact_y,
        'y_extension_ft': y_extension_ft,
        'y_extension_m': y_extension_m,
        'y_extension_in': y_extension_in,
        'serve_kmh': serve_kmh,
        'serve_mph': serve_mph,
        'peak_hip_vy_fts': peak_hip_vy_fts,
        'peak_hip_vy_mps': peak_hip_vy_mps,
        'peak_hip_vy_mph': peak_hip_vy_mph,
        'peak_hip_vy_time': peak_hip_vy_time,
        'peak_wrist_speed_fts': peak_wrist_speed_fts,
        'peak_wrist_speed_mps': peak_wrist_speed_mps,
        'peak_wrist_speed_mph': peak_wrist_speed_mph,
        'peak_wrist_speed_time': peak_wrist_speed_time,
        'dt': dt
    }

# -----------------------------
# Plotting functions (updated to use selected_players order)
# -----------------------------
def plot_hip_position(players_data, selected_players):
    fig, ax = plt.subplots(figsize=(12, 5))
    first_selected = True
    # Create a dict for fast lookup
    player_dict = {p['player_name']: p for p in players_data}
    for i, player_name in enumerate(selected_players):
        p = player_dict[player_name]
        line, = ax.plot(p['time'], p['hip_y_smooth'], label=p['player_name'])
        color = line.get_color()
        ax.axvline(p['trophy_time'], color=color, linestyle='--', alpha=0.7)
        ax.axvline(p['contact_time'], color=color, linestyle='--', alpha=0.7)
        
        if first_selected:
            ax.text(p['trophy_time']-0.1, 0.5*(p['trophy_y']), "Trophy pos",
                    fontsize=10, color='black', va='bottom', ha='center', rotation=90)
            ax.text(p['contact_time']+0.1, 0.5*(p['trophy_y']) , "Contact",
                    fontsize=10, color='black', va='bottom', ha='center', rotation=90)
            first_selected = False
        
        # Use i (index in selected_players) for vertical position
        ax.text(0.02, 0.90 - i*0.05, 
                f"{p['player_name']}: {p['y_extension_ft']:.2f} ft ({p['y_extension_in']:.1f} in, {p['y_extension_m']:.2f} m)",
                transform=ax.transAxes, fontsize=9,
                color=color, bbox=dict(facecolor='white', edgecolor=color, alpha=0.7))
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Y Position (ft)")
    ax.set_title("Hip Vertical Displacement")
    ax.grid(True)
    #ax.legend()
    return fig

def plot_hip_velocity(players_data, selected_players):
    fig, ax = plt.subplots(figsize=(12, 5))
    first_selected = True
    player_dict = {p['player_name']: p for p in players_data}
    for i, player_name in enumerate(selected_players):
        p = player_dict[player_name]
        line, = ax.plot(p['time'], p['hip_vy_fts'], label=p['player_name'])
        color = line.get_color()
        ax.axvline(p['trophy_time'], color=color, linestyle=':', alpha=0.7)
        ax.axvline(p['contact_time'], color=color, linestyle='--', alpha=0.7)

        if first_selected:
            ax.text(p['trophy_time']-0.1, p['trophy_y'] + 0.1, "Trophy pos",
                    fontsize=10, color='black', va='bottom', ha='center', rotation=90)
            ax.text(p['contact_time']+0.1, p['contact_y'] + 0.1, "Contact",
                    fontsize=10, color='black', va='bottom', ha='center', rotation=90)
            first_selected = False
        
        ax.text(0.02, 0.90 - i*0.10,
                f"{p['player_name']} Peak Vy:\n"
                f"{p['peak_hip_vy_fts']:.2f} ft/s | {p['peak_hip_vy_mps']:.2f} m/s | {p['peak_hip_vy_mph']:.2f} mph",
                transform=ax.transAxes, fontsize=9,
                color=color, bbox=dict(facecolor='white', edgecolor=color, alpha=0.7))
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Y Velocity (ft/s)")
    ax.set_title("Hip Vertical Velocity")
    ax.grid(True)
    #ax.legend()
    return fig

def plot_wrist_speed(players_data, selected_players):
    fig, ax = plt.subplots(figsize=(12, 5))
    first_selected = True
    player_dict = {p['player_name']: p for p in players_data}
    for i, player_name in enumerate(selected_players):
        p = player_dict[player_name]
        line, = ax.plot(p['time'], p['wrist_speed_fts'], label=p['player_name'])
        color = line.get_color()
        ax.axvline(p['contact_time'], color=color, linestyle='--', alpha=0.7)
        
        ax.text(0.02, 0.90 - i*0.10,
                f"{p['player_name']} Peak Speed:\n"
                f"{p['peak_wrist_speed_fts']:.2f} ft/s | {p['peak_wrist_speed_mps']:.2f} m/s | {p['peak_wrist_speed_mph']:.2f} mph",
                transform=ax.transAxes, fontsize=9,
                color=color, bbox=dict(facecolor='white', edgecolor=color, alpha=0.7))

        if first_selected:
            ax.text(p['contact_time']+0.1, p['contact_y'] + 0.1, "Contact",
                    fontsize=10, color='black', va='bottom', ha='center', rotation=90)
            first_selected = False
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wrist Speed (ft/s)")
    ax.set_title("Wrist Speed (2D Magnitude)")
    ax.grid(True)
    #ax.legend()
    return fig

# -----------------------------
# Helper to build full row for selected table
# -----------------------------
def full_row(p):
    return {
        "Player": p['player_name'],
        "Hip Disp (ft)": p['y_extension_ft'],
        "Hip Vel (mph)": p['peak_hip_vy_mph'],
        "Wrist Spd (mph)": p['peak_wrist_speed_mph'],
        "Serve (mph)": p['serve_mph']
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Tennis Serve AI Analysis", layout="wide")
st.title("🎾 Tennis Serve Biomechanics AI Analyzer")
st.markdown("Comparing professional tennis players' serve data")

# --- Load all .dat files ---
dat_files = glob.glob("*.dat")
if not dat_files:
    st.error("No .dat files found in the current directory.")
    st.stop()

with st.spinner(f"Processing {len(dat_files)} file(s)..."):
    players_data = [process_dat_file(f) for f in dat_files]

# --- Initialise session state ---
if "selected_players" not in st.session_state:
    st.session_state.selected_players = set()
    st.session_state.selected_players ={
        "FedererFlatWidec",
        "leoAdFlat006crop",
        "ChadAdFlat001crop"
    }
if "last_selected_player" not in st.session_state:
    st.session_state.last_selected_player = "ChadAdFlat001crop" #None
if "scroll_trigger" not in st.session_state:
    st.session_state.scroll_trigger = False

# --- Build data for the two tables ---
selected_list = [p for p in players_data if p['player_name'] in st.session_state.selected_players]
unselected_list = [p for p in players_data if p['player_name'] not in st.session_state.selected_players]



# Selected table: full metrics with a "❌" checkbox column for removal
df_selected = pd.DataFrame([full_row(p) for p in selected_list])
if not df_selected.empty:
    df_selected.insert(0, "❌", False)

# Unselected table: only player name + "+" checkbox
df_unselected = pd.DataFrame([{"+": False, "Player": p['player_name']} for p in unselected_list])



# -----------------------------
# Video playback (with auto-scroll)
# -----------------------------
video_player = st.session_state.last_selected_player
if video_player:
    video_container_id = "video-container"
    st.markdown(f'<div id="{video_container_id}"></div>', unsafe_allow_html=True)
    
    st.subheader(f"Video: {video_player}")
    video_path = os.path.join("videos", f"{video_player}annotated.mp4")
    if not os.path.exists(video_path):
        for ext in ['.mov', '.avi', '.mkv', '.webm']:
            alt_path = os.path.join("videos", f"{video_player}{ext}")
            if os.path.exists(alt_path):
                video_path = alt_path
                break
    if os.path.exists(video_path):
        st.video(video_path)
    else:
        st.info(f"No video found for {video_player} in the 'videos/' folder.")
    
    if st.session_state.scroll_trigger:
        st.markdown(f"""
        <script>
            var element = document.getElementById("{video_container_id}");
            if(element) {{
                element.scrollIntoView({{ behavior: "smooth", block: "start" }});
            }}
        </script>
        """, unsafe_allow_html=True)
        st.session_state.scroll_trigger = False
else:
    st.info("Select at least one player to see their video.")

# -----------------------------
# Images below video (trophy & contact)
# -----------------------------
if video_player:
    st.markdown("**Angles at trophy position and contact**")
    col1, col2 = st.columns(2)

    img1_path = os.path.join("videos", f"{video_player}trophy.png")
    img2_path = os.path.join("videos", f"{video_player}contact.png")

    for ext in ['.png', '.jpeg', '.jpg']:
        if not os.path.exists(img1_path):
            img1_path = os.path.join("images", f"{video_player}_1{ext}")
        if not os.path.exists(img2_path):
            img2_path = os.path.join("images", f"{video_player}_2{ext}")

    with col1:
        if os.path.exists(img1_path):
            st.image(img1_path, caption=f"{video_player} - trophy", use_container_width=True)
        else:
            st.info(f"No image of trophy position for {video_player}")

    with col2:
        if os.path.exists(img2_path):
            st.image(img2_path, caption=f"{video_player} - contact", use_container_width=True)
        else:
            st.info(f"No image of contact angle for {video_player}")
else:
    st.info("Select a player to view their trophy and contact images.")



# -----------------------------
# Two-table selection interface with 75/25 ratio
# -----------------------------
st.subheader("Select players for comparison and see data below")

col_left, col_right = st.columns([3, 1])   # 75% / 25% split

with col_left:
    st.markdown("#### ✅ Selected")
    if not df_selected.empty:
        edited_selected = st.data_editor(
            df_selected,
            column_config={
                "❌": st.column_config.CheckboxColumn("❌", default=False),
                "Player": st.column_config.TextColumn("Player", disabled=True),
                "Hip Disp (ft)": st.column_config.NumberColumn("Hip Disp (ft)", format="%.2f"),
                "Hip Vel (mph)": st.column_config.NumberColumn("Hip Vel (mph)", format="%.2f"),
                "Wrist Spd (mph)": st.column_config.NumberColumn("Wrist Spd (mph)", format="%.2f"),
                "Serve (mph)": st.column_config.NumberColumn("Serve (mph)", format="%.2f"),
            },
            hide_index=True,
            use_container_width=True,
            key="editor_selected"
        )
        for _, row in edited_selected.iterrows():
            if row["❌"] and row["Player"] in st.session_state.selected_players:
                st.session_state.selected_players.discard(row["Player"])
                if st.session_state.last_selected_player == row["Player"]:
                    st.session_state.last_selected_player = next(iter(st.session_state.selected_players), None)
                st.session_state.scroll_trigger = True
    else:
        st.info("No players selected. Use the '+' on the right.")

with col_right:
    st.markdown("#### 📋 Available")
    if not df_unselected.empty:
        edited_unselected = st.data_editor(
            df_unselected,
            column_config={
                "+": st.column_config.CheckboxColumn("+", default=False, width="small"),
                "Player": st.column_config.TextColumn("Player", disabled=True),
            },
            hide_index=True,
            use_container_width=True,
            key="editor_unselected"
        )
        for _, row in edited_unselected.iterrows():
            if row["+"] and row["Player"] not in st.session_state.selected_players:
                st.session_state.selected_players.add(row["Player"])
                st.session_state.last_selected_player = row["Player"]
                st.session_state.scroll_trigger = True
    else:
        st.info("All players are selected.")

# Ensure last_selected_player is valid
if st.session_state.selected_players and st.session_state.last_selected_player not in st.session_state.selected_players:
    st.session_state.last_selected_player = next(iter(st.session_state.selected_players))

selected_players = list(st.session_state.selected_players)
st.caption(f"**{len(selected_players)} player(s) selected**")



# -----------------------------
# Plots
# -----------------------------
if selected_players:
    st.subheader("Hip Displacement")
    st.pyplot(plot_hip_position(players_data, selected_players))
    
    st.subheader("Hip Vertical Velocity")
    st.pyplot(plot_hip_velocity(players_data, selected_players))
    
    st.subheader("Wrist Speed")
    st.pyplot(plot_wrist_speed(players_data, selected_players))
else:
    st.warning("Select at least one player to display plots.")