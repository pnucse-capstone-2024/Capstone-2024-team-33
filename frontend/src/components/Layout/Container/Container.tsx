import { ReactNode } from 'react';

interface ContainerProps {
  children: ReactNode;
}

const Container = ({ children }: ContainerProps) => (
  <div className="bg-background flex min-h-[calc(100vh-200px)] w-full justify-center pt-[62px]">
    <div className="flex w-full max-w-[1000px] flex-col gap-10 py-12">
      {children}
    </div>
  </div>
);

export default Container;
