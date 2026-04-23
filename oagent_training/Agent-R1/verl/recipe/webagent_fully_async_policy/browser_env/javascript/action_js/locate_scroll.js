(params) => {
    function scrollAtPoint({x, y, direction, distance}) {
        // 1. 检查 direction 是否传入，不传直接报错
        if (!direction) {
            throw new Error('direction 参数是必需的！');
        }
        const validDirections = ['up', 'down', 'left', 'right'];
        if (!validDirections.includes(direction)) {
            throw new Error(`direction 必须是以下值之一: ${validDirections.join(', ')}`);
        }
        function isScrollable(el, axis) {
            if (!(el instanceof HTMLElement)) return false;
            const s = getComputedStyle(el);
            if (axis === 'vertical') return (['auto','scroll'].includes(s.overflowY)) && el.scrollHeight > el.clientHeight;
            return (['auto','scroll'].includes(s.overflowX)) && el.scrollWidth > el.clientWidth;
        }
        function findScrollable(x, y, axis) {
            let el = document.elementFromPoint(x, y);
            while (el) {
                if (isScrollable(el, axis)) return el;
                el = el.parentElement;
            }
            return document.documentElement;
        }
        const axis = (direction === 'up' || direction === 'down') ? 'vertical' : 'horizontal';
        const scrollOnPage = (x === undefined || y === undefined);
        if (distance === undefined) {
            if (axis === 'vertical') {
                distance = window.innerHeight;
            } else {
                distance = window.innerWidth;
            }
        }
        let c;
        if (scrollOnPage) {
            c = document.documentElement;
        } else {
            c = findScrollable(x, y, axis);
        }
        if (c === document.documentElement) {
            if (axis === 'vertical') {
                window.scrollBy({ top: (direction === 'down' ? distance : -distance), behavior: 'smooth' });
            } else {
                window.scrollBy({ left: (direction === 'right' ? distance : -distance), behavior: 'smooth' });
            }
        } else {
            if (axis === 'vertical') {
                c.scrollTop += (direction === 'down' ? distance : -distance);
            } else {
                c.scrollLeft += (direction === 'right' ? distance : -distance);
            }
        }
    }
    scrollAtPoint(params);
    return true;
}