import Milestone1 from "./milestone1.mdx";
import Milestone2 from "./milestone2.mdx";

export default function Milestones() {
  return (
    <div className="mdx">
      <Milestone1 />
      <hr
        style={{
          margin: "4rem 0",
          border: "none",
          borderTop: "1px solid #e5e5e5",
        }}
      />
      <Milestone2 />
    </div>
  );
}
