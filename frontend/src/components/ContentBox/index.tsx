import { ReactNode } from 'react';

interface ContentBoxProps {
  title?: string;
  children: ReactNode;
  titleCenter?: boolean;
}

const ContentBox = ({ title, children, titleCenter }: ContentBoxProps) => (
  <div className="flex w-full flex-col gap-8 rounded-lg bg-white px-10 py-8 shadow-md">
    {title && (
      <p className={`${titleCenter && 'text-center'} text-xl font-bold`}>
        {title}
      </p>
    )}
    {children}
  </div>
);

export default ContentBox;
