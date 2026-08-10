import { useEffect, useState } from "react";

type AuthMode = "login" | "register";

type AuthDialogProps = {
  open: boolean;
  user: { username: string; avatar?: string | null } | null;
  loading: boolean;
  error: string;
  message: string;
  onClose: () => void;
  onLogin: (username: string, password: string) => Promise<void>;
  onRegister: (username: string, password: string) => Promise<void>;
  onChangePassword: (oldPassword: string, newPassword: string) => Promise<void>;
};

export function AuthDialog({ open, user, loading, error, message, onClose, onLogin, onRegister, onChangePassword }: AuthDialogProps) {
  const [mode, setMode] = useState<AuthMode>("register");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [oldPassword, setOldPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");

  useEffect(() => {
    if (!open) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [open, onClose]);

  if (!open) return null;

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (mode === "login") await onLogin(username.trim(), password);
    else await onRegister(username.trim(), password);
  }

  async function handlePasswordSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await onChangePassword(oldPassword, newPassword);
    setOldPassword("");
    setNewPassword("");
  }

  return (
    <div className="auth-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="auth-dialog" role="dialog" aria-modal="true" aria-labelledby="auth-title">
        <button className="auth-close" type="button" onClick={onClose} aria-label="Close sign in dialog">×</button>
        <div className="auth-brand"><img src="/favicon.svg" alt="" /><span>Smart Developer</span></div>
        <div className="auth-heading">
          <p className="eyebrow">Secure workspace</p>
          <h2 id="auth-title">{user ? "Account security" : mode === "login" ? "Welcome back" : "Create your account"}</h2>
          <p>{user ? `Signed in as ${user.username}. Update your password securely below.` : mode === "login" ? "Sign in with the username and password stored in your PostgreSQL database." : "Your password is hashed by the FastAPI backend before it is stored."}</p>
        </div>
        {!user && <><div className="auth-tabs" aria-label="Authentication mode">
          <button type="button" className={mode === "register" ? "active" : ""} onClick={() => setMode("register")}>Create account</button>
          <button type="button" className={mode === "login" ? "active" : ""} onClick={() => setMode("login")}>Sign in</button>
        </div>
        <form onSubmit={handleSubmit}>
          <label>Username<input autoFocus value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" placeholder="Enter your username" maxLength={50} required /></label>
          <label>Password<input value={password} onChange={(event) => setPassword(event.target.value)} autoComplete={mode === "login" ? "current-password" : "new-password"} type="password" placeholder="Enter your password" maxLength={72} required /></label>
          {error && <div className="auth-error" role="alert">{error}</div>}
          <button className="auth-submit" type="submit" disabled={loading || !username.trim() || !password}>{loading ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}</button>
        </form></>}
        {user && <form onSubmit={handlePasswordSubmit}>
          <div className="account-user-summary"><span className="user-avatar">{user.avatar ? <img src={user.avatar} alt="" /> : user.username.slice(0, 2).toUpperCase()}</span><div><strong>{user.username}</strong><small>Authenticated account</small></div></div>
          <label>Current password<input autoFocus value={oldPassword} onChange={(event) => setOldPassword(event.target.value)} autoComplete="current-password" type="password" placeholder="Enter current password" maxLength={72} required /></label>
          <label>New password<input value={newPassword} onChange={(event) => setNewPassword(event.target.value)} autoComplete="new-password" type="password" placeholder="At least 6 characters" minLength={6} maxLength={72} required /></label>
          {error && <div className="auth-error" role="alert">{error}</div>}
          {message && <div className="auth-success" role="status">{message}</div>}
          <button className="auth-submit" type="submit" disabled={loading || !oldPassword || newPassword.length < 6}>{loading ? "Updating…" : "Update password"}</button>
        </form>}
        <p className="auth-security"><span aria-hidden="true">●</span> Credentials are sent only to your FastAPI API.</p>
      </section>
    </div>
  );
}
