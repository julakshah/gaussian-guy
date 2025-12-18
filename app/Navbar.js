import Link from 'next/link'


export default function Navbar() {
return (
<nav className="navbar">
<div className="nav-inner">
<span className="logo">MyProject</span>
<div className="links">
<Link href="/">Home</Link>
<Link href="/milestones/milestone1">Milestone 1</Link>
<Link href="/milestones/milestone2">Milestone 2</Link>
</div>
</div>
</nav>
)
}
