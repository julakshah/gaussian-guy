import Link from "next/link";

export default function Navbar() {
  return (
    <nav className="navbar">
      <div className="nav-inner">
        <img
          className="arm-icon"
          src="site-elements/robot-arm.png"
          alt="robot arm icon by verry purnomo - Flaticon"
          width="50"
          height="50"
        />
        <div className="links">
          <Link href="/">Home</Link>
          <Link href="/milestones">Milestones</Link>
          <Link href="/setup">Project Setup</Link>
          <Link href="/arm">Arm Movement</Link>
          <Link href="/pathing">Arm Path Generation</Link>
          <Link href="/processing">Image Processing</Link>
          <Link href="/reflection">Next Steps & Reflection</Link>
        </div>
      </div>
    </nav>
  );
}
