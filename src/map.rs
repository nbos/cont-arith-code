use std::fmt;
use std::iter::FromIterator;

/// A Map implementation without insertion or deletion. Implements `Eq +
/// Ord \+ Hash`, O(log(n)) lookups, O(n*log(n)) construction, O(n)
/// unions/merges.
#[derive(Debug,Clone,PartialEq,Eq,PartialOrd,Ord,Hash)]
pub struct Map<K, V>(pub Vec<(K, V)>);

impl<K, V> fmt::Display for Map<K, V>
where
    K: fmt::Display,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let tuples: Vec<String> = self.0.iter()
            .map(|(k, v)| format!("({},{:?})", k, v))
            .collect();
        write!(f, "[{}]", tuples.join(","))
    }
}

impl<K, V> Map<K, V> {
    /// Creates an empty map
    pub fn empty() -> Self {
        Map(Vec::new())
    }

    /// Creates a map with a single key-value pair
    pub fn singleton(key: K, value: V) -> Self {
        Map(vec![(key, value)])
    }

    pub fn len(&self) -> usize {
	self.0.len()
    }

    /// Looks up a key using binary search
    pub fn lookup(&self, key: &K) -> Option<&V>
    where K: Clone + Ord
    {
	self.index_of(key).map(|i| self.get_value(i))
    }

    pub fn index_of(&self, key: &K) -> Option<usize>
    where K: Clone + Ord
    {
        self.0.binary_search_by_key(key, |(k, _)| k.clone()).ok()
    }

    pub fn get_key(&self, i: usize) -> &K {
	&self.0[i].0
    }

    pub fn get_value(&self, i: usize) -> &V {
	&self.0[i].1
    }

    /// Map over the values of the Map
    pub fn map<F,U>(&self, mut f: F) -> Map<K,U>
    where
	F: FnMut(&V) -> U,
	K: Clone
    {
	Map(self.iter().map(|(k,v)| (k.clone(),f(v))).collect())
    }

    /// Returns an iterator over references to the key-value pairs
    pub fn iter(&self) -> std::slice::Iter<'_, (K, V)> {
        self.0.iter()
    }

    /// Returns an iterator over mutable references to the key-value pairs
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, (K, V)> {
        self.0.iter_mut()
    }

    /// Returns an iterator over references to the values
    pub fn values(&self) ->
	std::iter::Map<std::slice::Iter<'_, (K, V)>, fn(&(K, V)) -> &V>
    {
        self.0.iter().map(|(_, v)| v)
    }

    /// Returns an iterator over references to the keys
    pub fn keys(&self) ->
	std::iter::Map<std::slice::Iter<'_, (K, V)>, fn(&(K, V)) -> &K>
    {
        self.0.iter().map(|(k, _)| k)
    }

    /// Returns an iterator over mutable references to the values
    pub fn values_mut(&mut self) ->
	std::iter::Map< std::slice::IterMut<'_, (K, V)>,
			fn(&mut (K, V)) -> &mut V >
    {
        self.0.iter_mut().map(|(_, v)| v)
    }
}

impl<K, V> IntoIterator for Map<K, V> {
    type Item = (K, V);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<K: Ord + Eq, V> FromIterator<(K, V)> for Map<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut vec: Vec<(K, V)> = iter.into_iter().collect();
        vec.sort_by(|a, b| a.0.cmp(&b.0));
        vec.dedup_by(|a, b| a.0 == b.0);
        Map(vec)
    }
}
