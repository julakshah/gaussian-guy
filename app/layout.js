import "./globals.css";
import Navbar from "./Navbar";

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <title>Gaussian Guy</title>
      </head>
      <body>
        <Navbar />
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
