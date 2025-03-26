package student;
/**
 * This class is for creating the pile of cards
 * that players play onto.
 *
 * @author Paul Liao
 */
public class CardPile {

    /** The size of the pile. */
    private int size = 1;
    /** The card on the top of pile. */
    private Card top;

    /**
     * Constructor that create a new card pile.
     *
     * @param topCard This is the top card in the card pile.
     */
    public CardPile(Card topCard) {
        top = topCard;
    }

    /**
     * To check if the card been chosen is valid.
     *
     * @param card This is the card been chosen.
     * @return boolean This returns the result of valid.
     */
    public boolean canPlay(Card card) {
        if (card == null) {
            return false;
        }
        return top.getRankNum() <= card.getRankNum()
                || top.getSuitNum() == card.getSuitNum();
    }

    /**
     * To put the card been chosen on the top of card pile.
     *
     * @param card This is the card been chosen.
     */
    public void play(Card card) {
        if (canPlay(card)) {
            top = card;
            size++;
        } else {
            System.out.println("Illegal move detected!");
        }
    }

    /**
     * It gets the size of the pile.
     *
     * @return int This returns the size of the pile.
     */
    public int getNumCards() {
        return size;
    }

    /**
     * It gets the top card.
     *
     * @return Card This returns the top card.
     */
    public Card getTopCard() {
        return top;
    }
}
