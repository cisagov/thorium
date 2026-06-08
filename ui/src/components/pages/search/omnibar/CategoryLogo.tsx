import { FaArrowUp19, FaFolderOpen, FaMagnifyingGlass, FaPen, FaPeopleGroup } from 'react-icons/fa6';
import { FaClock, FaTag } from 'react-icons/fa';

type CategoryLogoProps = {
  category: string;
};

export const CategoryLogo: React.FC<CategoryLogoProps> = ({ category }) => {
  switch (category) {
    case 'group':
      return <FaPeopleGroup title={category} />;
    case 'index':
      return <FaFolderOpen title={category} />;
    case 'text':
      return <FaPen title={category} />;
    case 'tag':
    case 'hide tags':
      return <FaTag title={category} />;
    case 'limit':
      return <FaArrowUp19 title={category} />;
    case 'time':
      return <FaClock title={category} />;
    case '':
      return <></>;
    default:
      return <FaMagnifyingGlass title={category} />;
  }
};
