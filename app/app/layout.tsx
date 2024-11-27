// app/layout.tsx
"use client";
import "bootstrap/dist/css/bootstrap.min.css";
import { ReactNode } from "react";
import { motion } from "framer-motion";

// Layout component serves as a wrapper for all pages
export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head />
      <body>
        <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{
          duration: 0.5,
          ease: "easeInOut",
        }}
        className="container">{children}</motion.div>
      </body>
    </html>
  );
}
