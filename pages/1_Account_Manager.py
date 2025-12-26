import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import streamlit as st

# Load config
with open('config/auth_config.yaml', 'r') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initialise authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

st.title('Account Manager')
# --- Tabs for Login & Register ---
login_tab, register_tab = st.tabs(["🔐 Login", "📝 Register"])

# ---------------------- LOGIN TAB ----------------------
with login_tab:
    try:
        authenticator.login(
            location='main',
            key='login_form',
            max_login_attempts=3,
            captcha=True
        )

        if st.session_state.get('authentication_status'):
            st.success(f"Welcome *{st.session_state.get('name')}*. You can now access the chatbot.")
            authenticator.logout(button_name="Logout", key="logout_button")

        elif st.session_state.get('authentication_status') is False:
            st.error("Username/password is incorrect")
            

        else:
            st.info("Please enter your username and password")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# ---------------------- REGISTER TAB ----------------------
with register_tab:
    with st.form("custom_registration_form"):
        new_email = st.text_input("Email")
        new_username = st.text_input("Username")
        new_first_name = st.text_input("First Name")
        new_last_name = st.text_input("Last Name")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
    
        # Role selection dropdown
        role = st.selectbox(
            "Select Role",
            options=["doctor", "nurse"],
            help="Choose your role in the system"
        )
    
        submit = st.form_submit_button("Register")
    
        if submit:
            # Validation
            if new_password != confirm_password:
                st.error("Passwords do not match")
            elif new_username in config["credentials"]["usernames"]:
                st.error("Username already exists")
            elif not all([new_email, new_username, new_first_name, new_last_name, new_password]):
                st.error("All fields are required")
            else:
                # Hash the password
                hashed_password = stauth.Hasher.hash(new_password)
            
                # Add user to config with selected role
                config["credentials"]["usernames"][new_username] = {
                    "email": new_email,
                    "first_name": new_first_name,
                    "last_name": new_last_name,
                    "password": hashed_password,
                    "logged_in": False,
                    "roles": [role]  # Use the selected role
                }
                # Save updated config
                with open("config/auth_config.yaml", "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
            
                st.success(f"User {new_username} registered successfully with role: {role}")
                st.info("You can now log in with your credentials")