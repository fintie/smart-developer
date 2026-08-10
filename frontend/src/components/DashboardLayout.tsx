import type { ReactNode } from "react";

export type SignedInUser = { id: number; username: string; bio?: string | null; avatar?: string | null };

type DashboardLayoutProps = {
  children: ReactNode;
  user: SignedInUser | null;
  onOpenAuth: () => void;
  onSignOut: () => void;
  activePage: "dashboard" | "collection";
  collectionCount: number;
  onOpenCollection: () => void;
  onOpenDashboard: () => void;
};

const navigation = [
  { href: "#search", icon: "⌕", label: "Site Search" },
];

export function DashboardLayout({ children, user, onOpenAuth, onSignOut, activePage, collectionCount, onOpenCollection, onOpenDashboard }: DashboardLayoutProps) {
  const initials = user?.username.slice(0, 2).toUpperCase() || "GU";

  function goHome() {
    window.location.hash = "search";
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  return (
    <main className="dashboard-page">
      <aside className="dashboard-sidebar">
        <button className="dashboard-logo" type="button" onClick={goHome} aria-label="Go to Smart Developer home">
          <img src="/favicon.svg" alt="Smart Developer" />
          <span><strong>Smart Developer</strong><small>Development intelligence</small></span>
        </button>
        <p className="nav-section-label">Workspace</p>
        <nav className="dashboard-navigation" aria-label="Primary navigation">
          {navigation.map((item, index) => <a key={item.href} onClick={onOpenDashboard} className={activePage === "dashboard" && index === 0 ? "active" : ""} href={item.href}><span aria-hidden="true">{item.icon}</span>{item.label}</a>)}
          {collectionCount > 0 && <button type="button" className={activePage === "collection" ? "active" : ""} onClick={onOpenCollection}><span aria-hidden="true">♡</span><span>Collection</span></button>}
        </nav>
        <button className="sidebar-user" type="button" onClick={onOpenAuth} aria-label={user ? "Open user account" : "Sign in"}>
          <span className="user-avatar">{user?.avatar ? <img src={user.avatar} alt="" /> : initials}</span>
          <span className="user-meta"><strong>{user?.username || "Guest user"}</strong><small>{user ? "Signed in" : "Click to sign in"}</small></span>
          <span aria-hidden="true">•••</span>
        </button>
      </aside>
      <section className="dashboard-main">
        <header className="dashboard-header">
          <div><p>Workspace</p><strong>Site Intelligence</strong></div>
          <div className="header-actions">
            <span className="live-status"><i />Platform online</span>
            <button className="header-avatar" type="button" onClick={onOpenAuth} aria-label={user ? `Open account for ${user.username}` : "Sign in"}>{user?.avatar ? <img src={user.avatar} alt="" /> : initials}</button>
            {user && <button className="signout-button" type="button" onClick={onSignOut}>Sign out</button>}
          </div>
        </header>
        <div className="dashboard-workspace">{children}</div>
      </section>
    </main>
  );
}
