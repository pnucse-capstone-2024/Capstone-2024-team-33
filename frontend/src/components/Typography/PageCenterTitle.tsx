interface PageCenterTitleProps {
  title: string;
}

const PageCenterTitle = ({ title }: PageCenterTitleProps) => (
  <div className="flex w-full justify-center">
    <div className="flex w-full max-w-[360px] items-center justify-center border-b-2 border-black pb-2 text-2xl font-bold">
      <p>{title}</p>
    </div>
  </div>
);

export default PageCenterTitle;
