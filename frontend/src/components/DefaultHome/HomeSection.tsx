import { Button } from '@mui/material';
import { ReactNode } from 'react';

interface HomeSectionProps {
  title: string;
  content: ReactNode;
  buttonText: string;
  buttonUrl: string;
  imageSource: string;
  imagePosition?: 'right' | 'left';
}

const HomeSection = ({
  title,
  content,
  buttonText,
  buttonUrl,
  imageSource,
  imagePosition,
}: HomeSectionProps) => {
  return (
    <div className="flex flex-col gap-16 rounded-lg bg-white px-20 py-16 shadow-md">
      <p className="text-center text-xl font-bold">{title}</p>
      <div
        className={`flex ${imagePosition === 'left' && 'flex-row-reverse'} justify-between`}
      >
        <div
          className={`flex flex-grow flex-col justify-between gap-8 ${imagePosition === 'left' && 'items-end text-right'} `}
        >
          <div className="leading-8">{content}</div>
          <Button
            variant="outlined"
            size="large"
            href={buttonUrl}
            className="w-[215px] rounded border py-3 font-bold"
          >
            {buttonText}
          </Button>
        </div>
        <img src={imageSource} className="h-[224px] w-[355px] rounded" />
      </div>
    </div>
  );
};

export default HomeSection;
