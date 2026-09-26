import { useEffect, useRef, useState, type ReactNode } from "react";

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
  activeSection: WorkspaceSection;
  onSelectSection: (section: WorkspaceSection) => void;
};

export type WorkspaceSection = "listings" | "opportunities" | "growth";

const navigation: Array<{ section: WorkspaceSection; icon: string; label: string }> = [
  { section: "listings", icon: "⌂", label: "Listings" },
  { section: "opportunities", icon: "⌕", label: "Opportunity Finder" },
  { section: "growth", icon: "↗", label: "Growth Map" },
];

export function DashboardLayout({ children, user, onOpenAuth, onSignOut, activePage, collectionCount, onOpenCollection, onOpenDashboard, activeSection, onSelectSection }: DashboardLayoutProps) {
  const initials = user?.username.slice(0, 2).toUpperCase() || "GU";
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);
  const accountMenuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function closeMenu(event: MouseEvent) {
      if (!accountMenuRef.current?.contains(event.target as Node)) setAccountMenuOpen(false);
    }
    document.addEventListener("mousedown", closeMenu);
    return () => document.removeEventListener("mousedown", closeMenu);
  }, []);

  function goHome() {
    onOpenDashboard();
    onSelectSection("listings");
    window.history.pushState(null, "", "#listings");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function selectSection(section: WorkspaceSection) {
    onOpenDashboard();
    onSelectSection(section);
    setAccountMenuOpen(false);
    window.history.pushState(null, "", `#${section}`);
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
          {navigation.map((item) => <button key={item.section} type="button" onClick={() => selectSection(item.section)} className={activePage === "dashboard" && activeSection === item.section ? "active" : ""}><span aria-hidden="true">{item.icon}</span><span>{item.label}</span></button>)}
          {collectionCount > 0 && <button type="button" className={activePage === "collection" ? "active" : ""} onClick={onOpenCollection}><span aria-hidden="true">♡</span><span>Collection</span></button>}
        </nav>
        <div className="account-menu-wrap sidebar-account" ref={accountMenuRef}>
        <button className="sidebar-user" type="button" onClick={() => setAccountMenuOpen((open) => !open)} aria-label="Open account and navigation menu" aria-expanded={accountMenuOpen} aria-haspopup="menu">
          <span className="user-avatar">{user?.avatar ? <img src={user.avatar} alt="" /> : initials}</span>
          <span className="user-meta"><strong>{user?.username || "Guest user"}</strong><small>{user ? "Signed in" : "Click to sign in"}</small></span>
          <span aria-hidden="true">•••</span>
        </button>
        {accountMenuOpen && <div className="account-menu" role="menu">
          <div className="account-menu-profile"><span className="user-avatar">{user?.avatar ? <img src={user.avatar} alt="" /> : initials}</span><div><strong>{user?.username || "Guest user"}</strong><small>{user ? "Signed in" : "Explore as guest"}</small></div></div>
          <button type="button" role="menuitem" onClick={() => selectSection("listings")}>Property Listings <span>⌂</span></button>
          <button type="button" role="menuitem" onClick={() => selectSection("opportunities")}>Opportunities Finder <span>⌕</span></button>
          <button type="button" role="menuitem" onClick={() => selectSection("growth")}>Growth Map <span>↗</span></button>
          {collectionCount > 0 && <button type="button" role="menuitem" onClick={() => { onOpenCollection(); setAccountMenuOpen(false); }}>Saved collection <span>{collectionCount}</span></button>}
          <div className="account-menu-divider" />
          <button type="button" role="menuitem" onClick={() => { setAccountMenuOpen(false); onOpenAuth(); }}>{user ? "Account settings" : "Sign in or create account"}<span>→</span></button>
          {user && <button className="account-menu-signout" type="button" role="menuitem" onClick={() => { setAccountMenuOpen(false); onSignOut(); }}>Sign out</button>}
        </div>}
        </div>
      </aside>
      <section className="dashboard-main">
        <header className="dashboard-header">
          <nav className="workspace-tabs" aria-label="Workspace sections">
            <button type="button" className={activePage === "dashboard" && activeSection === "listings" ? "active" : ""} onClick={() => selectSection("listings")}>Property Listings</button>
            <button type="button" className={activePage === "dashboard" && activeSection === "opportunities" ? "active" : ""} onClick={() => selectSection("opportunities")}>Opportunities Finder</button>
            <button type="button" className={activePage === "dashboard" && activeSection === "growth" ? "active" : ""} onClick={() => selectSection("growth")}>Growth Map</button>
          </nav>
        </header>
        <div className="dashboard-workspace">{children}</div>
      </section>
    </main>
  );
}
