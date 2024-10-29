interface PageTitleProps {
  title: string;
}

const PageTitle = ({ title }: PageTitleProps) => (
  <div className="text-2xl font-bold">{title}</div>
);

export default PageTitle;
