interface CircleBadgeProps {
  title: string;
  content: string;
}

const CircleBadge = ({ title, content }: CircleBadgeProps) => (
  <div className="flex h-[105px] w-[105px] flex-col justify-center gap-1 rounded-full border border-primary text-center">
    <p className="text-xs">{title}</p>
    <p className="text-base font-bold">{content}</p>
  </div>
);

export default CircleBadge;
